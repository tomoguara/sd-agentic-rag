"""Self-Review: verifies a draft after generation (design D8).

The deterministic checks always run: every citation is in the turn's
grounded allow-list, at least `min_citations` citations, and the share of
cited sentences (sentences of 4+ words containing a citation marker) meets
`min_cited_sentence_ratio`. A detected refusal passes. In `llm` mode an
extra groundedness check runs only after the deterministic checks passed,
so it can fail a draft but never pass one (tighten-only).
"""

import re
from typing import List, Optional

from sd_agentic_rag.agent import prompts
from sd_agentic_rag.agent.config import SelfReviewConfig
from sd_agentic_rag.agent.llm import LLMCallError
from sd_agentic_rag.agent.state import GateVerdict
from sd_agentic_rag.agent.trace import TraceRecorder
from sd_agentic_rag.sd_agentic_ground import GroundedContext
from sd_agentic_rag.sd_agentic_guard import Guardrails


CITATION_RE = re.compile(r"\[CITATION:[^\]]+\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_CITATIONS_RE = re.compile(r"^(\s*\[CITATION:[^\]]+\])+")
MIN_SENTENCE_WORDS = 4


def cited_sentence_ratio(text: str) -> float:
    """Share of sentences (4+ words) that carry a citation marker.

    Markers written right after the full stop ("... fixed it. [CITATION:X]")
    belong to the preceding sentence. A draft with no countable sentence has
    ratio 1.0.
    """
    sentences: List[List[str]] = []          # [text, has_citation]
    for piece in _SENTENCE_SPLIT_RE.split(text or ""):
        piece = piece.strip()
        if not piece:
            continue
        leading = _LEADING_CITATIONS_RE.match(piece)
        if leading and sentences:
            sentences[-1][1] = True
            piece = piece[leading.end():].strip()
            if not piece:
                continue
        sentences.append([piece, bool(CITATION_RE.search(piece))])
    counted = [cited for sentence, cited in sentences
               if len(CITATION_RE.sub(" ", sentence).split()) >= MIN_SENTENCE_WORDS]
    if not counted:
        return 1.0
    return sum(counted) / len(counted)


class SelfReview:
    """Draft verification gate."""

    def __init__(self, config: Optional[SelfReviewConfig] = None, guardrails: Optional[Guardrails] = None,
                 llm_call=None, trace: Optional[TraceRecorder] = None):
        self.config = config or SelfReviewConfig()
        self.guardrails = guardrails or Guardrails()
        self.llm_call = llm_call
        self.trace = trace or TraceRecorder()

    def deterministic(self, draft: str, grounded: GroundedContext) -> GateVerdict:
        if self.guardrails.check_refusal(draft):
            return GateVerdict(gate="self_review", passed=True, mode="heuristic", reasons=["refusal"])
        reasons = []
        check = self.guardrails.verify_citations(draft, grounded.citations)
        if not check["valid"]:
            reasons.append(f"cites tickets not in the grounded evidence: {', '.join(check['missing'])}")
        count = len(check["answer_citations"])
        if count < self.config.min_citations:
            reasons.append(f"has {count} citation(s), at least {self.config.min_citations} required")
        ratio = cited_sentence_ratio(draft)
        if ratio < self.config.min_cited_sentence_ratio:
            reasons.append(f"only {ratio:.0%} of sentences are cited "
                           f"(min {self.config.min_cited_sentence_ratio:.0%})")
        return GateVerdict(gate="self_review", passed=not reasons, mode="heuristic", reasons=reasons)

    def check(self, draft: str, grounded: GroundedContext, budget=None, allow_llm: bool = True) -> GateVerdict:
        """Verdict for a draft. BudgetExceeded from the optional LLM check propagates."""
        verdict = self.deterministic(draft, grounded)
        refusal = verdict.reasons == ["refusal"]
        if verdict.passed and not refusal and allow_llm and self.config.mode == "llm" and self.llm_call:
            try:
                raw = self.llm_call(
                    system=prompts.SELF_REVIEW_SYSTEM,
                    user=prompts.SELF_REVIEW_USER.format(context=grounded.context_text, draft=draft),
                    schema={"unsupported_claims": list},
                    budget=budget,
                )
                claims = [str(c) for c in raw["unsupported_claims"] if str(c).strip()]
                if claims:
                    verdict = GateVerdict(gate="self_review", passed=False, mode="llm",
                                          reasons=[f"unsupported claim: {c}" for c in claims])
                else:
                    verdict.mode = "llm"
            except LLMCallError as e:
                # Tighten-only: an unusable LLM check leaves the deterministic verdict
                self.trace.record("self_review", "self_review_llm_error", {"error": str(e)})
        self.trace.record("self_review", "verdict", verdict.to_dict())
        return verdict
