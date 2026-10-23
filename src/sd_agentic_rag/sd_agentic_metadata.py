"""Chunk metadata fields shared by Ingest (Stage 1) and Store (Stage 3)."""


# Chunk metadata fields and the value stored when a ticket lacks one (design D3).
# Also the allow-list StorePipeline uses for the Qdrant payload.
METADATA_DEFAULTS = {
    "title": "",
    "service": "",
    "severity": "",
    "ticket_type": "",
    "status": "",
    "priority": "",
    "sla_minutes": None,
    "tags": [],
    "assigned_to": "",
    "created_at": "",
    "updated_at": "",
}


def metadata_default(field: str):
    """Return a fresh copy of the missing-value default for a metadata field."""
    default = METADATA_DEFAULTS[field]
    return list(default) if isinstance(default, list) else default
