"""Offline fakes for the agent tests: store, retriever, embedding service, LLM client.

Nothing here touches the network; every LLM call goes through the real
`json_call()` with a scripted client so budget accounting is exercised.
"""

import functools
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from sd_agentic_rag.agent.llm import json_call
from sd_agentic_rag.sd_agentic_generate import GeneratedAnswer
from sd_agentic_rag.sd_agentic_ground import ContextGrounding
from sd_agentic_rag.sd_agentic_guard import Guardrails


def make_ticket_point(point_id: str, ticket_id: str, text: str, chunk_index: int = 0,
                      vector: Optional[List[float]] = None, **metadata) -> Dict[str, Any]:
    payload = {"ticket_id": ticket_id, "chunk_index": chunk_index, "chunk_text": text, **metadata}
    return {"id": point_id, "payload": payload, "vector": vector or [1.0, 0.0, 0.0]}


DEFAULT_POINTS = [
    make_ticket_point("p1", "LAT001234", "API latency fixed by adding a composite index.",
                      vector=[1.0, 0.0, 0.0], service="Database Services", severity="High",
                      ticket_type="Performance Issue", status="Resolved", priority="P1",
                      sla_minutes=240, assigned_to="Database Team"),
    make_ticket_point("p2", "LAT002345", "Replication lag on DB01 fixed by tuning WAL settings.",
                      vector=[0.0, 1.0, 0.0], service="Database Services", severity="Medium",
                      ticket_type="Incident", status="Closed", priority="P2", sla_minutes=60),
    make_ticket_point("p3", "NET000111", "VPN drops resolved by renewing the certificate.",
                      vector=[0.0, 0.0, 1.0], service="Network", severity="Low",
                      ticket_type="Incident", status="Open", priority="P3"),
    make_ticket_point("p4", "LAT001234", "Follow-up: slow query log threshold lowered to 1s.",
                      chunk_index=1, vector=[0.9, 0.1, 0.0], service="Database Services",
                      severity="High", ticket_type="Performance Issue", status="Resolved",
                      priority="P1", sla_minutes=240, assigned_to="Database Team"),
]


class FakeQdrantClient:
    """The subset of QdrantClient the agent uses: scroll, retrieve, query_points."""

    def __init__(self, points: List[Dict[str, Any]]):
        self.points = points

    @staticmethod
    def _matches(point, flt) -> bool:
        if flt is None or not getattr(flt, "must", None):
            return True
        for condition in flt.must:
            if point["payload"].get(condition.key) != condition.match.value:
                return False
        return True

    def _record(self, point, with_vectors=False):
        return SimpleNamespace(id=point["id"], payload=dict(point["payload"]),
                               vector=point["vector"] if with_vectors else None)

    def scroll(self, collection_name, scroll_filter=None, limit=100, offset=None,
               with_payload=True, with_vectors=False):
        matching = [p for p in self.points if self._matches(p, scroll_filter)]
        start = offset or 0
        batch = matching[start:start + limit]
        next_offset = start + limit if start + limit < len(matching) else None
        return [self._record(p, with_vectors) for p in batch], next_offset

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [self._record(p, with_vectors) for p in self.points if p["id"] in ids]


class FakeVectorStore:
    collection_name = "tickets__fake"

    def __init__(self, points=None):
        self.points = points if points is not None else DEFAULT_POINTS
        self.client = FakeQdrantClient(self.points)
        self.searches = []

    def search(self, query_vector, top_k=10, filter_conditions=None, with_payload=True,
               with_vector=False, score_threshold=None):
        self.searches.append({"filter_conditions": filter_conditions, "top_k": top_k})
        results = []
        for p in self.points:
            if filter_conditions and any(p["payload"].get(c.key) != c.match.value for c in filter_conditions):
                continue
            score = sum(a * b for a, b in zip(query_vector, p["vector"]))
            results.append({"id": p["id"], "score": score, "payload": dict(p["payload"])})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]


class FakeEmbeddingService:
    """Keyword-based 3-d embeddings: database / replication / network."""

    model = "fake-embedding"

    def __init__(self, vectors: Optional[Dict[str, List[float]]] = None):
        self.vectors = vectors or {}
        self.calls = []

    def embed_query(self, text: str) -> List[float]:
        self.calls.append(text)
        if text in self.vectors:
            return self.vectors[text]
        lower = text.lower()
        return [
            1.0 if ("latency" in lower or "slow" in lower or "index" in lower) else 0.0,
            1.0 if ("replica" in lower or "replication" in lower or "db01" in lower) else 0.0,
            1.0 if ("vpn" in lower or "network" in lower) else 0.0,
        ]

    embed_text = embed_query


class FakeRetriever:
    """HybridRetriever stand-in returning vector-store results (no BM25 by default)."""

    def __init__(self, vector_store: FakeVectorStore, embedding_service: FakeEmbeddingService,
                 bm25_index=None, known_values=None):
        from sd_agentic_rag.sd_agentic_retrieve import HybridRetriever
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_index = bm25_index
        self.vector_top_k = 20
        self.bm25_top_k = 20
        self.k_rrf = 60
        self.calls = []
        self._known = known_values
        self.rrf_fusion = functools.partial(HybridRetriever.rrf_fusion, self)

    def retrieve(self, query, top_k=10, query_embedding=None, vector_top_k=None, bm25_top_k=None):
        self.calls.append({"query": query, "top_k": top_k})
        if query_embedding is None:
            query_embedding = self.embedding_service.embed_query(query)
        vector_results = [r for r in self.vector_store.search(query_embedding, top_k=vector_top_k or 20)
                          if r["score"] > 0]
        bm25_results = self.bm25_index.search(query, top_k=bm25_top_k or 20) if self.bm25_index else []
        return self.rrf_fusion(vector_results, bm25_results, k=self.k_rrf)[:top_k]

    def known_metadata_values(self):
        if self._known is not None:
            return self._known
        from sd_agentic_rag.sd_agentic_retrieve import collect_known_values
        return collect_known_values([p["payload"] for p in self.vector_store.points])


class FakeGenerator:
    """Scripted AnswerGenerator: returns answers in order (last one repeats)."""

    def __init__(self, answers: List[str], tokens: int = 100):
        self.answers = list(answers)
        self.tokens = tokens
        self.calls = []

    def generate_answer(self, question, context, citations, system_prompt=None):
        self.calls.append({"question": question, "context": context, "system_prompt": system_prompt})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return GeneratedAnswer(answer=answer, citations=citations, prompt_tokens=self.tokens // 2,
                               completion_tokens=self.tokens // 2, total_tokens=self.tokens, model="fake")


class ScriptedChatClient:
    """chat.completions.create stub: each response is a dict (sent as JSON), a str, or an Exception."""

    def __init__(self, responses: List[Any], tokens: int = 50):
        self.responses = list(responses)
        self.tokens = tokens
        self.requests: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise RuntimeError("no scripted response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        content = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=self.tokens),
        )

    @property
    def prompts(self) -> List[str]:
        return [m["content"] for r in self.requests for m in r["messages"]]


def llm_from(client: ScriptedChatClient):
    """A json_call bound to a scripted client."""
    return functools.partial(json_call, client=client, model="stub-model")


def make_guardrails(**overrides) -> Guardrails:
    params = dict(enable_relevance_gate=False, min_tokens=1, enforce=True)
    params.update(overrides)
    return Guardrails(**params)


class FakePipeline:
    """SdAgenticRAGPipeline stand-in with real Ground and Guard stages."""

    def __init__(self, answers: Optional[List[str]] = None, points=None, guardrails=None,
                 embedding_service=None, max_passages: int = 5, max_tokens: int = 4000):
        self.embed_service = embedding_service or FakeEmbeddingService()
        self.vector_store = FakeVectorStore(points)
        self.hybrid_retriever = FakeRetriever(self.vector_store, self.embed_service)
        self.rerank_pipeline = None
        self.rerank_enabled = False
        self.ground_pipeline = ContextGrounding(max_passages=max_passages, max_tokens=max_tokens,
                                                min_relevance_score=0.05)
        self.generate_pipeline = FakeGenerator(answers or ["Fixed by adding an index. [CITATION:LAT001234]"])
        self.guard_pipeline = guardrails or make_guardrails()
        self.observability = None
