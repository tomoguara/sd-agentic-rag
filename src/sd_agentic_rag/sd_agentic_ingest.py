import csv
import json
import re
from typing import List, Dict, Optional

import tiktoken
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sd_agentic_rag.sd_agentic_metadata import metadata_default


class SdAgenticIngest:
    def __init__(self,
                 chunk_size: int=1000,
                 chunk_overlap: int=150,
                 chunk_type: str = "recursive", # "recursive" or "token"
                 chunk_header: Optional[bool] = None # None = use ingest.chunk_header from config
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_type = chunk_type
        self.chunk_header = chunk_header


    # load the tickets from the ticket source as a JSON or CSV file
    def load_ticket(self,
                    data_path: str,
                    format: str = "json"
    ) -> List[Dict]:
        file_format = (format or "").lower()
        lower_path = data_path.lower()
        is_json = lower_path.endswith(".json") or (
            file_format == "json" and not lower_path.endswith((".csv", ".tsv"))
        )

        if is_json:
            with open(data_path, "r", encoding="utf-8") as tickets:
                payload = json.load(tickets)

            if isinstance(payload, dict):
                rows = payload.get("tickets", [])
            else:
                rows = payload

            normalized_rows = []
            for ticket in rows:
                row = dict(ticket)
                user = row.pop("user", {})
                row["user_name"] = user.get("name", "")
                row["user_email"] = user.get("email", "")
                row["user_team"] = user.get("team", "")
                row["user_location"] = user.get("location", "")
                normalized_rows.append(row)
            return normalized_rows

        with open(data_path, "r", newline="", encoding="utf-8") as tickets:
            content = csv.DictReader(tickets)
            return list(content)


    # apply guardrails - Do this for PII's only - will extend later
    def clean_text(self,
                   text: str
    ) -> str:
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),  # SSN (if applicable)
            (r'\b\d{16}\b', '[CARD_REDACTED]'),  # Credit card-like patterns
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
        ]

        for pattern, replacement in pii_patterns:
            text = re.sub(pattern, replacement, text)

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Create  appropriate chunks
    def chunk_text(self,
                   text: str,
                   strategy: str=""
    ) -> List[str]:
        # Use instance chunk_type if no strategy provided
        if strategy == "":
            strategy = self.chunk_type

        if strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", "", " "]
            )
            chunks = splitter.split_text(text)
            return chunks
        elif strategy == "token":
            # Token-based chunking (requires tiktoken)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunks.append(encoding.decode(chunk_tokens))
            return chunks
        elif strategy == "semantic":
            raise NotImplementedError("Semantic chunking requires embedding model")
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


    # apply a simple deduplication strategy - exact string deduplication
    def dedupe_chunks(self, chunks: List[str]) -> List[str]:
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)

        return unique_chunks

    #  get the chunk and the metadata
    def process_tickets(self,
                    data_path: str,
                    format: str = "json",
                    clean: bool = True
    ) -> List[Dict]:
        chunks = []
        tickets = self.load_ticket(data_path, format)
        for ticket in tickets:
            # Combine description and resolution_notes for better searchability
            ticket_text = ticket.get("description", "")
            if ticket.get("resolution_notes") and ticket["resolution_notes"].strip():
                ticket_text += "\n\nResolution: " + ticket["resolution_notes"]

            # Clean the text before chunking (removes PII, normalizes whitespace)
            text_to_chunk = self.clean_text(ticket_text) if clean else ticket_text  # clean up the text
            description_chunk = self.chunk_text(text_to_chunk)  # chunk the text
            description_chunk = self.dedupe_chunks(description_chunk) # dedupe the chunks

            # Free-text fields that go into the header and metadata get the same PII cleaning
            def clean_field(value: str) -> str:
                value = str(value).strip()
                return self.clean_text(value) if clean and value else value

            title = clean_field(ticket.get("title", ""))
            tags_val = ticket.get("tags")
            # Tags can be a list (JSON) or a string (CSV); keep the original shape in
            # metadata, and store [] when missing or empty
            if isinstance(tags_val, list):
                tags_meta = [clean_field(t) for t in tags_val if t]
                tags = ", ".join(tags_meta)
            elif tags_val is None or str(tags_val).strip() == "":
                tags_meta = metadata_default("tags")
                tags = ""
            else:
                tags_meta = clean_field(tags_val)
                tags = tags_meta

            # sla_minutes is None when missing or empty (e.g. a blank CSV cell)
            sla_minutes = ticket.get("sla_minutes")
            if sla_minutes is None or str(sla_minutes).strip() == "":
                sla_minutes = metadata_default("sla_minutes")

            # Build chunk header from non-volatile fields if enabled
            chunk_header = ""
            use_header = self.chunk_header
            if use_header is None:
                use_header = self._get_ingest_config().get("chunk_header", True)
            if use_header:
                service = clean_field(ticket.get("service", ""))
                ticket_type = clean_field(ticket.get("ticket_type", ""))
                # Build clean header without volatile fields (status, priority, severity, SLA, assignee, dates)
                # Labelled so each value is unambiguous; empty fields are left out (design D3)
                header_fields = [("Title", title), ("Service", service), ("Type", ticket_type), ("Tags", tags)]
                header_parts = [f"{label}: {value}" for label, value in header_fields if value]
                if header_parts:
                    chunk_header = f"{' | '.join(header_parts)}\n\n"

            for i, chunk_text in enumerate(description_chunk): # attach metadata
                # Prepend header to chunk text
                final_chunk_text = chunk_header + chunk_text

                the_chunk = {
                    "ticket_id": ticket.get("ticket_id", ""),
                    "chunk_text": final_chunk_text,
                    "chunk_index": i,
                    "metadata": {
                        "title": title,
                        "service": ticket.get("service", ""),
                        "severity": ticket.get("severity", ""),
                        "ticket_type": ticket.get("ticket_type", ""),
                        "status": ticket.get("status", ""),
                        "priority": ticket.get("priority", ""),
                        "sla_minutes": sla_minutes,
                        "tags": tags_meta,
                        "assigned_to": ticket.get("assigned_to", ""),
                        "created_at": ticket.get("created_at", ""),
                        "updated_at": ticket.get("updated_at", "")
                    }
                }
                chunks.append(the_chunk)

                print(f"Chunk {i+1}/{len(description_chunk)} for ticket {ticket.get('ticket_id', '')}")


        return chunks

    def _get_ingest_config(self) -> Dict:
        """Get ingest configuration, either from Config singleton or defaults."""
        try:
            from sd_agentic_rag.sd_agentic_embed import Config
            return Config().get_ingest_config()
        except Exception:
            return {"chunk_header": True}

    def get_chunks(self,
                   data_path: str,
                   format: str = "json",
                   clean: bool = True
    ) -> List[Dict]:
        return self.process_tickets(data_path, format=format, clean=clean)


def main():
    data_path = "./data/synthetic_tickets.json"
    ingest = SdAgenticIngest(chunk_size=100, chunk_overlap=10)
    tickets = ingest.process_tickets(data_path, clean=True)

if __name__ == "__main__":
    main()