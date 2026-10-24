"""Agent memory in one local SQLite file (design D12).

- Thread conversation: each turn's question and final answer under a thread
  ID; `new_conversation()` starts a fresh epoch, hiding (not deleting) the
  earlier turns from the Planner.
- Step audit: every loop step, append-only. There is deliberately no API to
  update or delete a step.
- Past resolutions: questions whose answers passed every gate, with their
  cited ticket IDs and a float32 question vector tied to the embedding model.
  They are Planner hints only and are never evidence.

PII is redacted from everything before it is written.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sd_agentic_rag.sd_agentic_guard import Guardrails


SCHEMA_VERSION = 1
DB_ENV_VAR = "SD_AGENTIC_MEMORY_DB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    conversation_epoch INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    turn_no INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    cited_ticket_ids TEXT NOT NULL,
    partial INTEGER NOT NULL,
    refused INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_thread ON turns (thread_id, epoch, turn_no);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    turn_no INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    component TEXT NOT NULL,
    event TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS steps_thread ON steps (thread_id, turn_no, seq);
CREATE TABLE IF NOT EXISTS resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    question TEXT NOT NULL,
    cited_ticket_ids TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS resolutions_model ON resolutions (embedding_model);
"""


def resolve_db_path(cli_path: Optional[str] = None, config_path: Optional[str] = None) -> str:
    """`--db` > $SD_AGENTIC_MEMORY_DB > agent.memory.db_path."""
    if cli_path:
        return cli_path
    env_path = os.getenv(DB_ENV_VAR)
    if env_path:
        return env_path
    if config_path:
        return config_path
    from sd_agentic_rag.agent.config import MemoryConfig
    return MemoryConfig().db_path


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class AgentMemory:
    """Threads, turns, append-only steps and past resolutions in one SQLite file."""

    def __init__(self, path: str, redact: Optional[Callable[[str], str]] = None):
        self.path = str(path)
        self.redact = redact or Guardrails().redact_pii
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} has memory schema v{version}; this version supports v{SCHEMA_VERSION}")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    # -- threads ---------------------------------------------------------

    def _ensure_thread(self, thread_id: str) -> int:
        row = self._conn.execute(
            "SELECT conversation_epoch FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row:
            return row[0]
        now = _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO threads (thread_id, conversation_epoch, created_at, updated_at) VALUES (?, 0, ?, ?)",
                (thread_id, now, now),
            )
        return 0

    def current_epoch(self, thread_id: str) -> int:
        return self._ensure_thread(thread_id)

    def thread_exists(self, thread_id: str) -> bool:
        return self._conn.execute("SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,)).fetchone() is not None

    def new_conversation(self, thread_id: str) -> int:
        """Start a fresh conversation in the thread; earlier turns stay stored."""
        epoch = self._ensure_thread(thread_id) + 1
        with self._conn:
            self._conn.execute(
                "UPDATE threads SET conversation_epoch = ?, updated_at = ? WHERE thread_id = ?",
                (epoch, _now(), thread_id),
            )
        return epoch

    def next_turn_no(self, thread_id: str) -> int:
        """Turn numbers increase across conversations, so audit steps stay unambiguous."""
        row = self._conn.execute(
            "SELECT MAX(turn_no) FROM ("
            " SELECT turn_no FROM turns WHERE thread_id = ?"
            " UNION ALL SELECT turn_no FROM steps WHERE thread_id = ?)",
            (thread_id, thread_id),
        ).fetchone()
        return (row[0] or 0) + 1

    # -- turns -----------------------------------------------------------

    @staticmethod
    def _turn(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "turn_no": row["turn_no"],
            "epoch": row["epoch"],
            "question": row["question"],
            "answer": row["answer"],
            "cited_ticket_ids": json.loads(row["cited_ticket_ids"]),
            "partial": bool(row["partial"]),
            "refused": bool(row["refused"]),
            "created_at": row["created_at"],
        }

    def recent_turns(self, thread_id: str, limit: int) -> List[Dict[str, Any]]:
        """The current conversation's last `limit` turns, oldest first."""
        if limit <= 0 or not self.thread_exists(thread_id):
            return []
        epoch = self.current_epoch(thread_id)
        rows = self._conn.execute(
            "SELECT * FROM turns WHERE thread_id = ? AND epoch = ? ORDER BY turn_no DESC LIMIT ?",
            (thread_id, epoch, limit),
        ).fetchall()
        return [self._turn(r) for r in reversed(rows)]

    def history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Every turn of the current conversation, oldest first."""
        if not self.thread_exists(thread_id):
            return []
        epoch = self.current_epoch(thread_id)
        rows = self._conn.execute(
            "SELECT * FROM turns WHERE thread_id = ? AND epoch = ? ORDER BY turn_no",
            (thread_id, epoch),
        ).fetchall()
        return [self._turn(r) for r in rows]

    def save_turn(
        self,
        thread_id: str,
        turn_no: int,
        question: str,
        answer: str,
        cited_ticket_ids: List[str],
        partial: bool = False,
        refused: bool = False,
        resolution: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write the turn (and its past resolution, if any) in one transaction.

        Args:
            resolution: Optional {"vector": [...], "embedding_model": str} to
                store the turn as a past resolution
        """
        epoch = self._ensure_thread(thread_id)
        now = _now()
        question = self.redact(question)
        with self._conn:
            self._conn.execute(
                "INSERT INTO turns (thread_id, epoch, turn_no, question, answer, cited_ticket_ids,"
                " partial, refused, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, epoch, turn_no, question, self.redact(answer),
                 json.dumps(list(cited_ticket_ids)), int(partial), int(refused), now),
            )
            if resolution is not None:
                vector = np.asarray(resolution["vector"], dtype=np.float32)
                self._conn.execute(
                    "INSERT INTO resolutions (thread_id, question, cited_ticket_ids, embedding_model, dims,"
                    " vector, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (thread_id, question, json.dumps(list(cited_ticket_ids)), resolution["embedding_model"],
                     int(vector.shape[0]), vector.tobytes(), now),
                )
            self._conn.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))

    # -- step audit (append-only) ------------------------------------------

    def append_step(self, thread_id: str, turn_no: int, step: Dict[str, Any]) -> None:
        """Append one TraceRecorder event to the audit."""
        payload = self.redact(json.dumps(step.get("payload") or {}, default=str))
        with self._conn:
            self._conn.execute(
                "INSERT INTO steps (thread_id, turn_no, iteration, seq, component, event, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, turn_no, int(step.get("iteration") or 0), int(step.get("seq") or 0),
                 step.get("component", ""), step.get("event", ""), payload,
                 step.get("timestamp") or _now()),
            )

    def step_sink(self, thread_id: str, turn_no: int) -> Callable[[Dict[str, Any]], None]:
        """A TraceRecorder sink writing to this thread's audit."""
        return lambda step: self.append_step(thread_id, turn_no, step)

    def audit(self, thread_id: str) -> List[Dict[str, Any]]:
        """Every stored step of the thread across all turns and conversations, oldest first."""
        rows = self._conn.execute(
            "SELECT s.*, t.epoch AS epoch FROM steps s"
            " LEFT JOIN turns t ON t.thread_id = s.thread_id AND t.turn_no = s.turn_no"
            " WHERE s.thread_id = ? ORDER BY s.id",
            (thread_id,),
        ).fetchall()
        return [
            {
                "turn_no": r["turn_no"],
                "epoch": r["epoch"],
                "iteration": r["iteration"],
                "seq": r["seq"],
                "component": r["component"],
                "event": r["event"],
                "payload": json.loads(r["payload"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # -- past resolutions ------------------------------------------------

    def search_resolutions(
        self,
        vector: List[float],
        embedding_model: str,
        top_k: int = 3,
        min_similarity: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Most similar past resolutions recorded with the active embedding model."""
        query = np.asarray(vector, dtype=np.float32)
        rows = self._conn.execute(
            "SELECT * FROM resolutions WHERE embedding_model = ? AND dims = ?",
            (embedding_model, int(query.shape[0])),
        ).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        similarities = (matrix @ query) / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-similarities)
        results = []
        for i in order[:top_k]:
            if similarities[i] < min_similarity:
                break
            row = rows[int(i)]
            results.append({
                "question": row["question"],
                "cited_ticket_ids": json.loads(row["cited_ticket_ids"]),
                "thread_id": row["thread_id"],
                "similarity": float(similarities[i]),
            })
        return results
