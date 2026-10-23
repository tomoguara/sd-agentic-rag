import pytest
from sd_agentic_rag.sd_agentic_ingest import SdAgenticIngest

def test_load_tickets():
    data_path = "./data/synthetic_tickets.json"
    tickets = SdAgenticIngest().load_ticket(data_path)
    assert tickets is not None

@pytest.mark.parametrize(
        "text_with_pii, expected_redaction", [
            ("This is my SSN numbed: 123-45-6789", "[SSN_REDACTED]"),
            ("This is my email address numbed: test@example.com", "[EMAIL_REDACTED]"),
            ("This is my Credit card number: 1234567890123456", "[CARD_REDACTED]"),
            ("This is not a valid string", "[TEXT_REDACTED]")
        ]
)
def test_clean_text_remove(text_with_pii, expected_redaction):
    result = SdAgenticIngest().clean_text(text_with_pii)
    print(result)
    valid_redactions = [
        "[SSN_REDACTED]",
        "[EMAIL_REDACTED]",
        "[CARD_REDACTED]"
    ]
    if expected_redaction in valid_redactions:
        assert expected_redaction in result, f"Expected {expected_redaction} to be in result"
    else:
        assert expected_redaction not in result, f"Expected{expected_redaction} to not be in result"


def test_get_chunks():
    data_path = "./data/synthetic_tickets.json"
    ingest = SdAgenticIngest(chunk_size=100, chunk_overlap=10)
    tickets = ingest.get_chunks(data_path, clean=True)


def test_dedupe_chunks():
    """Test that exact duplicate chunks are removed, keeping only first occurrence."""
    ingest = SdAgenticIngest()
    chunks = ["first", "second", "first", "third", "second", "first"]
    result = ingest.dedupe_chunks(chunks)
    # Should keep first occurrences only
    assert result == ["first", "second", "third"]
    assert len(result) == 3



def _write_ticket(tmp_path, **fields):
    import json
    ticket = {
        "ticket_id": "INC900",
        "title": "Login failure for jane.doe@acme.com",
        "service": "Auth",
        "ticket_type": "Incident",
        "tags": ["login", "owner bob@acme.com"],
        "description": "User cannot log in.",
        "resolution_notes": "Reset the session.",
    }
    ticket.update(fields)
    path = tmp_path / "tickets.json"
    path.write_text(json.dumps([ticket]))
    return str(path)


def test_header_and_metadata_are_pii_cleaned(tmp_path, monkeypatch):
    """Title and tags are cleaned in the chunk header and in metadata."""
    ingest = SdAgenticIngest()
    monkeypatch.setattr(ingest, "_get_ingest_config", lambda: {"chunk_header": True})
    chunks = ingest.process_tickets(_write_ticket(tmp_path), clean=True)

    assert chunks
    for chunk in chunks:
        assert "@acme.com" not in chunk["chunk_text"]
        assert chunk["chunk_text"].startswith(
            "Title: Login failure for [EMAIL_REDACTED] | Service: Auth | Type: Incident"
            " | Tags: login, owner [EMAIL_REDACTED]\n\n"
        )
        assert chunk["metadata"]["title"] == "Login failure for [EMAIL_REDACTED]"
        assert chunk["metadata"]["tags"] == ["login", "owner [EMAIL_REDACTED]"]


def test_string_tags_are_pii_cleaned(tmp_path, monkeypatch):
    """Comma-separated tag strings are cleaned too."""
    ingest = SdAgenticIngest()
    monkeypatch.setattr(ingest, "_get_ingest_config", lambda: {"chunk_header": True})
    chunks = ingest.process_tickets(
        _write_ticket(tmp_path, tags="login, bob@acme.com"), clean=True
    )

    assert chunks[0]["metadata"]["tags"] == "login, [EMAIL_REDACTED]"
    assert "bob@acme.com" not in chunks[0]["chunk_text"]


def test_header_and_metadata_raw_without_clean(tmp_path, monkeypatch):
    """With clean=False the title is kept as-is."""
    ingest = SdAgenticIngest()
    monkeypatch.setattr(ingest, "_get_ingest_config", lambda: {"chunk_header": True})
    chunks = ingest.process_tickets(_write_ticket(tmp_path), clean=False)

    assert chunks[0]["metadata"]["title"] == "Login failure for jane.doe@acme.com"


def test_chunk_header_argument_overrides_config(tmp_path, monkeypatch):
    """chunk_header=False (--no-chunk-header) wins over ingest.chunk_header: true"""
    ingest = SdAgenticIngest(chunk_header=False)
    monkeypatch.setattr(ingest, "_get_ingest_config", lambda: {"chunk_header": True})
    chunks = ingest.process_tickets(_write_ticket(tmp_path), clean=True)

    assert not chunks[0]["chunk_text"].startswith("Title:")


def test_no_chunk_header_flag_reaches_ingest(monkeypatch):
    """`sd-agentic-rag ingest --no-chunk-header` passes chunk_header=False to run()"""
    import sys
    from sd_agentic_rag import cli
    from sd_agentic_rag.rag_commands import rag_ingest

    calls = {}

    def fake_run(self, **kwargs):
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(rag_ingest.RagIngestCommand, "run", fake_run)
    monkeypatch.setattr(rag_ingest.RagIngestCommand, "format_output", lambda self, r, f: "")

    for argv, expected in [(["--no-chunk-header"], False), ([], None)]:
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["sd-agentic-rag", "ingest", "-d", "x.json", *argv])
        with pytest.raises(SystemExit):
            cli.main()
        assert calls["chunk_header"] is expected


def test_ingest_result_has_collection_name(monkeypatch):
    """The ingest result names the collection, so the summary does not show N/A"""
    from unittest.mock import MagicMock
    import sd_agentic_rag
    from sd_agentic_rag.rag_commands.rag_ingest import RagIngestCommand

    store = MagicMock(collection_name="tickets_text_embedding_3_small")
    monkeypatch.setattr(sd_agentic_rag, "create_vector_store", MagicMock(return_value=store))
    monkeypatch.setattr(sd_agentic_rag, "SdAgenticIngest", MagicMock())
    monkeypatch.setattr(sd_agentic_rag, "EmbeddingService", MagicMock())
    embed_pipeline = MagicMock()
    embed_pipeline.return_value.process_chunks.return_value = []
    monkeypatch.setattr(sd_agentic_rag, "EmbeddingPipeline", embed_pipeline)
    monkeypatch.setattr(sd_agentic_rag, "StorePipeline", MagicMock())

    command = RagIngestCommand(debug=True)
    result = command.run("data.json")

    assert result["collection_name"] == "tickets_text_embedding_3_small"
    assert "tickets_text_embedding_3_small" in command.format_output(result)


def test_missing_tags_and_sla_use_d3_defaults(tmp_path, monkeypatch):
    """Missing tags -> [], missing sla_minutes -> None (design D3)"""
    import json
    ingest = SdAgenticIngest()
    monkeypatch.setattr(ingest, "_get_ingest_config", lambda: {"chunk_header": True})
    path = tmp_path / "tickets.json"
    path.write_text(json.dumps([{"ticket_id": "INC901", "title": "VPN drops", "description": "Drops hourly."}]))

    metadata = ingest.process_tickets(str(path), clean=True)[0]["metadata"]

    assert metadata["tags"] == []
    assert metadata["sla_minutes"] is None
    assert metadata["status"] == ""


def test_blank_csv_cells_use_d3_defaults(tmp_path, monkeypatch):
    """Blank CSV cells for tags / sla_minutes count as missing"""
    ingest = SdAgenticIngest()
    monkeypatch.setattr(ingest, "_get_ingest_config", lambda: {"chunk_header": True})
    path = tmp_path / "tickets.csv"
    path.write_text("ticket_id,title,description,tags,sla_minutes\nINC902,VPN drops,Drops hourly.,,\n")

    chunk = ingest.process_tickets(str(path), format="csv", clean=True)[0]

    assert chunk["metadata"]["tags"] == []
    assert chunk["metadata"]["sla_minutes"] is None
    assert "Tags:" not in chunk["chunk_text"]
