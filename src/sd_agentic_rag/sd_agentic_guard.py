"""Guard Wrapper for Stage 7: Generate - Enforce quality, safety, and traceability.

This module provides guardrails to wrap the Generate stage to enforce:
- Citation verification
- Relevance gate
- Refusal detection
- PII redaction
- Length constraints
- Safety filters
"""

import os
import re
from typing import List, Dict, Optional
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


STANDARD_REFUSAL = "I cannot find sufficient information in the ticket data to answer this question."

class Config:
    """Configuration manager for Guard Wrapper."""

    _instance = None
    _config = None

    def __new__(cls, config_path: str = "config/config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = cls._load_config(config_path)
        return cls._instance

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "guard": {
                    "enable_citation_check": True,
                    "enable_relevance_gate": True,
                    "enable_pii_redact": True,
                    "enable_safety_filter": False,
                    "min_citations_per_answer": 1,
                    "relevance_threshold": 0.5,
                    "max_tokens": 2000,
                    "min_tokens": 10
                }
            }

    def get(self, *keys, default=None):
        """Get config value using dot notation."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_guard_config(self) -> Dict:
        """Get guard-specific configuration."""
        return self._config.get("guard", {})


class Guardrails:
    """Guardrails wrapper for answer generation.

    This class implements Guard Wrapper for the SdAgentic RAG system.
    It wraps the answer generation to enforce quality, safety, and traceability.
    """

    def __init__(
        self,
        enable_citation_check: bool = True,
        enable_relevance_gate: bool = True,
        enable_pii_redact: bool = True,
        enable_safety_filter: bool = False,
        min_citations_per_answer: int = 1,
        relevance_threshold: float = 0.5,
        max_tokens: int = 2000,
        min_tokens: int = 10,
        embedding_service=None,
        enforce: bool = True
    ):
        """
        Initialize guardrails.

        Args:
            enable_citation_check: Verify citations exist in context
            enable_relevance_gate: Check answer relevance to question
            enable_pii_redact: Redact PII from answers
            enable_safety_filter: Block harmful content
            min_citations_per_answer: Minimum citations required
            relevance_threshold: Minimum similarity for relevance check
            max_tokens: Maximum answer length in tokens
            min_tokens: Minimum answer length in tokens
            embedding_service: Optional embedding service for relevance check
        """
        self.enable_citation_check = enable_citation_check
        self.enable_relevance_gate = enable_relevance_gate
        self.enable_pii_redact = enable_pii_redact
        self.enable_safety_filter = enable_safety_filter
        self.min_citations = min_citations_per_answer
        self.relevance_threshold = relevance_threshold
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.embedding_service = embedding_service
        self.enforce = enforce
        self.enforced = False
        self.replaced = False

    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),  # SSN
            (r'\b\d{16}\b', '[CARD_REDACTED]'),  # Credit card
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE_REDACTED]'),  # Phone
        ]

        for pattern, replacement in pii_patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def extract_citations(self, text: str) -> List[str]:
        """Extract citation markers from text."""
        pattern = r'\[CITATION:([^\]]+)\]'
        return re.findall(pattern, text)

    def verify_citations(
        self,
        answer: str,
        context_citations: List[Dict]
    ) -> Dict:
        """
        Verify that all citations in answer exist in context.
        """
        answer_citations = self.extract_citations(answer)
        context_ids = {c["ticket_id"] for c in context_citations}

        missing = [
            cit for cit in answer_citations
            if cit not in context_ids
        ]

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "answer_citations": answer_citations,
            "context_citations": list(context_ids)
        }

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using the embedding service."""
        if self.embedding_service:
            try:
                return self.embedding_service.embed_text(text)
            except Exception:
                pass
        return None

    def check_relevance(
        self,
        question: str,
        answer: str,
        embedding_service=None
    ) -> Dict:
        """
        Check if answer is relevant to question using cosine similarity.
        """
        service = embedding_service or self.embedding_service

        if not service:
            return {
                "relevant": True,
                "similarity": 1.0,
                "threshold": self.relevance_threshold,
                "warning": "No embedding service available"
            }

        try:
            q_emb = self._get_embedding(question)
            a_emb = self._get_embedding(answer)

            if q_emb is None or a_emb is None:
                return {
                    "relevant": True,
                    "similarity": 1.0,
                    "threshold": self.relevance_threshold,
                    "warning": "Failed to get embeddings"
                }

            dot_product = sum(q * a for q, a in zip(q_emb, a_emb))
            q_norm = sum(q * q for q in q_emb) ** 0.5
            a_norm = sum(a * a for a in a_emb) ** 0.5

            if q_norm == 0 or a_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (q_norm * a_norm)

            return {
                "relevant": similarity >= self.relevance_threshold,
                "similarity": similarity,
                "threshold": self.relevance_threshold
            }
        except Exception:
            return {
                "relevant": True,
                "similarity": 1.0,
                "threshold": self.relevance_threshold,
                "error": "Embedding check failed"
            }

    # Refusal phrases, matched only at the start of the answer (after an optional
    # apology) so support text that merely contains "cannot find" is not a refusal.
    REFUSAL_PATTERN = re.compile(
        r"^(?:(?:i'?m |i am )?sorry[,.]?\s*|unfortunately,?\s*)?"
        r"(?:i(?:'m|\s+am)?\s+)?"
        r"(?:cannot answer|cannot find|can't answer|can't find"
        r"|no sufficient|not enough information"
        r"|don't know|do not know|unable"
        r"|not applicable|outside my knowledge)"
    )

    def check_refusal(self, answer: str) -> bool:
        """Check if answer is a refusal (refusal phrase at the start of the answer)."""
        normalized = answer.strip().lower().replace("\u2019", "'")
        return bool(self.REFUSAL_PATTERN.match(normalized))

    def check_length(self, answer: str) -> Dict:
        """Check answer length constraints."""
        token_count = len(answer) // 4

        return {
            "valid": self.min_tokens <= token_count <= self.max_tokens,
            "token_count": token_count,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens
        }

    def check_safety(self, answer: str) -> Dict:
        """Check answer for harmful content using pattern matching."""
        harmful_patterns = [
            r"violence",
            r"self-harm",
            r"suicide",
            r"hate.*speech",
            r"discriminat",
            r"illegal",
            r"harass",
            r"threat",
        ]

        answer_lower = answer.lower()
        found_patterns = [
            pattern for pattern in harmful_patterns
            if re.search(pattern, answer_lower)
        ]

        return {
            "safe": len(found_patterns) == 0,
            "found_patterns": found_patterns
        }

    def process_answer(
        self,
        answer: str,
        question: str,
        citations: List[Dict],
        embedding_service=None
    ) -> Dict:
        """
        Apply all guardrails to answer.
        """
        result = {
            "allowed": True,
            "actions": [],
            "modified_answer": answer,
            "checks": {}
        }

        # Check refusal first for exemption logic
        is_refusal = self.check_refusal(answer)
        result["checks"]["refusal"] = {"detected": is_refusal}
        if is_refusal:
            result["actions"].append("Refusal detected")

        # 1. Citation check
        if self.enable_citation_check:
            citation_check = self.verify_citations(answer, citations)
            result["checks"]["citation"] = citation_check
            # Citation verification still applies even to refusals
            if not citation_check["valid"]:
                result["allowed"] = False
                result["actions"].append(f"Missing citations: {citation_check['missing']}")

        # 2. Citation count check (skip if refusal detected)
        answer_citations = self.extract_citations(answer)
        citation_count_check = {
            "count": len(answer_citations),
            "min_required": self.min_citations,
            "valid": len(answer_citations) >= self.min_citations
        }
        result["checks"]["citation_count"] = citation_count_check
        # Refusals skip citation count requirement
        if not citation_count_check["valid"] and not is_refusal:
            result["allowed"] = False
            result["actions"].append(
                f"Not enough citations: {len(answer_citations)} < {self.min_citations}"
            )

        # 3. Length check (skip if refusal detected)
        length_check = self.check_length(answer)
        result["checks"]["length"] = length_check
        # Refusals skip length requirement
        if not length_check["valid"] and not is_refusal:
            result["allowed"] = False
            result["actions"].append(
                f"Length out of bounds: {length_check['token_count']} tokens "
                f"(min: {self.min_tokens}, max: {self.max_tokens})"
            )

        # 4. Relevance check
        if self.enable_relevance_gate:
            relevance_check = self.check_relevance(question, answer, embedding_service)
            result["checks"]["relevance"] = relevance_check
            if not relevance_check["relevant"]:
                result["allowed"] = False
                result["actions"].append(
                    f"Low relevance: {relevance_check['similarity']:.3f} < "
                    f"{self.relevance_threshold}"
                )

        # 5. Safety check
        if self.enable_safety_filter:
            safety_check = self.check_safety(answer)
            result["checks"]["safety"] = safety_check
            if not safety_check["safe"]:
                result["allowed"] = False
                result["actions"].append(
                    f"Harmful content detected: {safety_check['found_patterns']}"
                )

        # 6. PII redaction (last - applies to final output)
        if self.enable_pii_redact:
            result["modified_answer"] = self.redact_pii(answer)
            if result["modified_answer"] != answer:
                result["actions"].append("PII redacted")

        # 7. Enforcement - if not allowed and enforce is True, substitute with refusal.
        # Refusals are exempt only from the count/length checks above, so a refusal
        # that fails citation verification or safety is replaced like any answer.
        # `enforced` reports whether enforcement was active; `replaced` whether it fired.
        result["enforced"] = self.enforce
        result["replaced"] = not result["allowed"] and self.enforce
        if result["replaced"]:
            result["modified_answer"] = STANDARD_REFUSAL
            result["actions"].append("Response blocked and replaced with standard refusal")

        return result


# config.yaml `guard:` keys that map 1:1 onto Guardrails constructor arguments
GUARD_CONFIG_KEYS = (
    "enforce",
    "enable_citation_check",
    "enable_relevance_gate",
    "enable_pii_redact",
    "enable_safety_filter",
    "min_citations_per_answer",
    "relevance_threshold",
    "max_tokens",
    "min_tokens",
)


def create_guardrails(**kwargs) -> Guardrails:
    """Create a Guardrails instance from config.yaml's `guard:` section.

    Values come from `guard:` (falling back to the Guardrails defaults);
    keyword arguments that are not None override them.
    """
    try:
        guard_config = Config().get_guard_config() or {}
    except Exception:
        guard_config = {}

    params = {key: guard_config[key] for key in GUARD_CONFIG_KEYS if key in guard_config}
    params.update({key: value for key, value in kwargs.items() if value is not None})
    return Guardrails(**params)


# Backward compatibility aliases
SdAgenticGuardrails = Guardrails
SdAgenticGuard = Guardrails
