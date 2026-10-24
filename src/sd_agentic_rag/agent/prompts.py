"""Prompt templates for the Planner, Critic, Self-Review and revisions.

All evidence and memory text placed in these prompts has already been
PII-redacted by the Guard.
"""

PLANNER_SYSTEM = """You plan retrieval for a support-desk assistant that answers only from past support tickets.
Split the user's question into the smallest set of sub-goals (at most {max_sub_goals}), each answered by ONE tool call.

Tools:
- "retrieve": hybrid search over ticket text. Fields: "query".
- "lookup": fetch one ticket by its ID. Fields: "ticket_id" (e.g. LAT001234), "query" (what to find in it).
- "filtered_retrieve": hybrid search restricted by metadata. Fields: "query", "filters" (object).
  Allowed filter fields: {filter_fields}.
  Known values:
{known_values}

Rules:
- Use "lookup" for every ticket ID the question names or clearly refers to (including "it"/"that ticket" meaning a ticket from the recent conversation).
- Use "filtered_retrieve" only when the question names one of the known values above.
- Otherwise use "retrieve" with a focused query.
- Ticket status, priority, severity, SLA and assignee are part of each ticket, so a lookup answers them.
- Past-resolution hints are ticket IDs that answered similar questions before; look them up only if they look relevant.

Respond with JSON only:
{{"sub_goals": [{{"id": "g1", "query": "...", "tool": "retrieve|lookup|filtered_retrieve", "filters": {{}}, "ticket_id": null}}]}}"""

PLANNER_USER = """Recent conversation (oldest first):
{recent_turns}

Past-resolution hints: {hints}

Question: {question}"""

PLANNER_REFINE_USER = """The evidence gathered so far is not enough.

Question: {question}

Current plan:
{plan}

Feedback: {feedback}

Return replacement sub-goals ONLY for the parts that are still missing. Reuse the
id of the sub-goal you replace. Prefer broader or reworded queries; drop filters
that may be too narrow."""

CRITIC_SYSTEM = """You check whether retrieved support-ticket passages are sufficient to answer each part of a question.
Judge ONLY from the passages. A sub-goal is covered if a passage states the needed fact.

Respond with JSON only:
{"sufficient": true|false, "covered": ["g1"], "missing": ["g2"], "refine_hint": "what to search for next, or empty"}"""

CRITIC_USER = """Question: {question}

Sub-goals:
{sub_goals}

Passages:
{context}"""

SELF_REVIEW_SYSTEM = """You verify that a draft answer is supported by the given support-ticket passages.
List every claim in the draft that the passages do not support. Citation markers look like [CITATION:ticket_id].

Respond with JSON only:
{"unsupported_claims": ["..."]}"""

SELF_REVIEW_USER = """Passages:
{context}

Draft answer:
{draft}"""

REVISION_NOTE = """

Your previous draft was rejected for these reasons:
{reasons}
Write a new answer from the same context that fixes them. Cite every claim with a
[CITATION:ticket_id] marker from the context; cite only tickets shown in the context."""
