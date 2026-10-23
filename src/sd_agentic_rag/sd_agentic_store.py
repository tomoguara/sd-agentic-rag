"""Stage 3: Storage - Vector Database Integration

Persists vectors + metadata in Qdrant for fast top-k similarity search.
Supports filtered queries by metadata (e.g., specific service or date range).
"""
import os
import uuid
from typing import List, Dict, Optional
from pathlib import Path

import yaml
from tqdm import tqdm

from sd_agentic_rag.model_selector import ModelSelector
from sd_agentic_rag.sd_agentic_metadata import METADATA_DEFAULTS, metadata_default


class Config:
    """Configuration manager for SdAgentic RAG system."""

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
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "batch_size": 100
                },
                "ingest": {
                    "chunk_size": 1000,
                    "chunk_overlap": 150,
                    "chunk_strategy": "recursive",
                    "clean_pii": True
                },
                "embed": {
                    "batch_size": 100,
                    "output_directory": "data/embedded"
                },
                "store": {
                    "qdrant": {
                        "host": "localhost",
                        "port": 6333,
                        "collection_name": "tickets",
                        "metrics": False
                    }
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

    def get_qdrant_config(self) -> Dict:
        """Get Qdrant-specific configuration."""
        return self._config.get("store", {}).get("qdrant", {})

def _slug_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe slug.
    
    Replaces runs of non-alphanumeric characters with a single hyphen.
    Lowercases the result.
    
    Examples:
        "text-embedding-3-small" -> "text-embedding-3-small"
        "qwen3-embedding:latest" -> "qwen3-embedding-latest"
        "GPT-3.5 Turbo" -> "gpt-3-5-turbo"
    """
    import re
    # Replace runs of non-alphanumeric characters with a single hyphen
    slug = re.sub(r'[^a-z0-9]+', '-', model_name.lower())
    # Remove leading/trailing hyphens
    return slug.strip('-') if slug else 'unknown'


def _make_collection_name(prefix: str, model_name: str) -> str:
    """Create collection name from prefix and model name.
    
    Returns format: prefix__model-slug
    """
    slug = _slug_model_name(model_name)
    return f"{prefix}__{slug}"


class VectorStore:
    """Vector store using Qdrant for storage and similarity search.

    Provides:
    - Persistent vector storage
    - Fast top-k similarity search
    - Metadata filtering (service, severity, etc.)
    - Upsert operations for updates
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        skip_init: bool = False
    ):
        """Initialize Qdrant client and create/retrieve collection.

        Args:
            host: Qdrant host (default: localhost from config)
            port: Qdrant port (default: 6333 from config)
            collection_name: Name of the vector collection (default: tickets)
            dimensions: Embedding dimensions (default: 1536 from model)
            skip_init: If True, skip collection creation (for testing)
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self.config = Config()
        qdrant_config = self.config.get_qdrant_config()

        self.host = host or qdrant_config.get("host", "localhost")
        self.port = port or qdrant_config.get("port", 6333)
        
        # Determine effective collection name
        # An explicit collection_name argument bypasses suffixing
        if collection_name:
            self.collection_name = collection_name
        else:
            # Use prefix from config with model name suffix
            prefix = qdrant_config.get("collection_name", "tickets")
            from sd_agentic_rag.model_selector import ModelSelector
            model_name = ModelSelector().get_embedding_model()
            self.collection_name = _make_collection_name(prefix, model_name)
        self.dimensions = dimensions or get_embedding_model_dimensions()

        # Initialize Qdrant client
        self.client = QdrantClient(host=self.host, port=self.port)

        # Create collection if it doesn't exist (unless skip_init is True)
        if not skip_init:
            if self.collection_exists():
                # Check vector size matches
                self._verify_dimensions()
            else:
                self._create_collection()

    def _create_collection(self):
        """Create a new vector collection in Qdrant."""
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimensions,
                distance=Distance.COSINE
            )
        )
        print(f"Created Qdrant collection: {self.collection_name}")
        
        # Create keyword payload indexes for ticket_id, service, severity, ticket_type, status
        for field_name in ["ticket_id", "service", "severity", "ticket_type", "status"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"  Created payload index for '{field_name}'")

    def _verify_dimensions(self) -> None:
        """Verify the collection's vector size matches the expected dimensions.
        
        The Qdrant CollectionInfo object is a Pydantic model. The vector configuration
        is accessed via: config.params.vectors.size
        
        Raises:
            RuntimeError: If the collection exists but has different dimensions.
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            # Qdrant CollectionInfo uses Pydantic model structure
            actual_dims = collection_info.config.params.vectors.size
            
            if actual_dims != self.dimensions:
                raise RuntimeError(
                    f"Collection '{self.collection_name}' has {actual_dims} dimensions, "
                    f"but model expects {self.dimensions} dimensions. "
                    f"Run 'ingest --wipe' to recreate the collection with the correct dimensions."
                )
        except Exception as e:
            # Re-raise RuntimeError as-is, wrap others
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(
                f"Failed to verify dimensions for collection '{self.collection_name}': {e}"
            )

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def upsert_points(self, points: List[Dict], batch_size: int = 100) -> None:
        """Upsert (insert/update) vectors into the store.

        Args:
            points: List of points, each with:
                - id: Unique identifier
                - vector: Embedding vector
                - payload: Metadata dictionary
            batch_size: Number of points to upsert in each batch
        """
        from qdrant_client.models import PointStruct

        if not points:
            print("No points to upsert")
            return

        print(f"Upserting {len(points)} points to Qdrant...")

        for i in tqdm(range(0, len(points), batch_size), desc="Upserting batches"):
            batch = points[i:i + batch_size]

            # Build payload: use existing payload if present, otherwise extract from chunk fields
            qdrant_points = [
                PointStruct(
                    id=point.get("id") or str(uuid.uuid4()),
                    vector=point.get("vector") or point.get("embedding", []),
                    payload=point.get("payload", {
                        "ticket_id": point.get("ticket_id", ""),
                        "chunk_index": point.get("chunk_index", 0),
                        "chunk_text": point.get("chunk_text", "")
                    })
                )
                for point in batch
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )

        print(f"Successfully upserted {len(points)} points")
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[List] = None,
        with_payload: bool = True,
        with_vector: bool = False,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Search for top-k most similar vectors.

        Args:
            query_vector: The embedding vector to search with
            top_k: Number of results to return
            filter_conditions: Optional list of Qdrant Filter conditions
            with_payload: Whether to include metadata in results
            with_vector: Whether to include the vector in results
            score_threshold: Optional minimum score threshold for results

        Returns:
            List of results with id, score, and payload (if requested)
        """
        from qdrant_client.models import Filter

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(must=filter_conditions) if filter_conditions else None,
            limit=top_k,
            with_payload=with_payload,
            score_threshold=score_threshold
        )

        # query_points returns QueryResponse object with points attribute
        points_list = search_result.points if hasattr(search_result, 'points') else search_result
        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload if with_payload else None
            }
            for point in points_list
        ]

    def filtered_search(
        self,
        query_vector: List[float],
        service: Optional[str] = None,
        severity: Optional[str] = None,
        ticket_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Search with common metadata filters.

        Args:
            query_vector: The embedding vector to search with
            service: Filter by service name
            severity: Filter by severity level
            ticket_type: Filter by ticket type
            top_k: Number of results to return

        Returns:
            List of filtered search results
        """
        from qdrant_client.models import FieldCondition, MatchValue

        conditions = []

        if service:
            conditions.append(
                FieldCondition(
                    key="service",
                    match=MatchValue(value=service)
                )
            )

        if severity:
            conditions.append(
                FieldCondition(
                    key="severity",
                    match=MatchValue(value=severity)
                )
            )

        if ticket_type:
            conditions.append(
                FieldCondition(
                    key="ticket_type",
                    match=MatchValue(value=ticket_type)
                )
            )

        return self.search(query_vector, top_k=top_k, filter_conditions=conditions)

    def get_all_points(self) -> List[Dict]:
        """Retrieve all points from the collection.
        
        Returns:
            List of all points with id, score, and payload
        """
        points = []
        offset = None
        
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True
            )
            batch, offset = result
            
            for point in batch:
                points.append({
                    "id": point.id,
                    "score": 0.0,  # Score is not available from scroll
                    "payload": point.payload
                })
            
            if offset is None:
                break
        
        return points

    def get_point(self, point_id: str) -> Optional[Dict]:
        """Retrieve a single point by ID.

        Args:
            point_id: The ID of the point to retrieve

        Returns:
            Point with id, vector, and payload, or None if not found
        """
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vector=True
            )
            if point:
                return {
                    "id": point[0].id,
                    "vector": point[0].vector,
                    "payload": point[0].payload
                }
            return None
        except Exception:
            return None

    def delete_point(self, point_id: str) -> bool:
        """Delete a point by ID.

        Args:
            point_id: The ID of the point to delete

        Returns:
            True if deletion was successful
        """
        from qdrant_client.models import PointId

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[PointId(id=point_id)]
            )
            return True
        except Exception:
            return False

    def count_points(self) -> int:
        """Count total number of points in the collection."""
        return self.client.count(
            collection_name=self.collection_name,
            exact=True
        ).count

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False



    def wipe(self) -> Dict[str, Any]:
        """Drop the collection and recreate it empty with the current dimensions.

        Recreating (rather than deleting points) also fixes a vector-size
        mismatch after the embedding model or its *_DIMM override changed.
        Returns statistics about the operation.
        """
        try:
            before_count = 0
            if self.collection_exists():
                try:
                    before_count = self.count_points()
                except Exception:
                    before_count = 0
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"Deleted collection '{self.collection_name}' ({before_count} points)")

            self._create_collection()
            after_count = self.count_points()

            return {
                "success": True,
                "before_count": before_count,
                "deleted_count": before_count - after_count,
                "after_count": after_count,
                "dimensions": self.dimensions,
                "collection": self.collection_name
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection": self.collection_name
            }

def get_embedding_model_dimensions() -> int:
    """Get embedding dimensions based on model configuration.
    
    Resolves dimensions in this order:
    1. The active provider's override (OLLAMA_EMBEDDING_MODEL_DIMM or
       OPENAI_EMBEDDING_MODEL_DIMM, from .env / sd-agentic-rag.cfg)
    2. Known model map from the active model (via model_selector)
    3. One-time probe if model is unknown

    Returns:
        Dimension count for embeddings

    Raises:
        ValueError: If the override is set but is not a positive integer
    """
    # Import dynamically to allow mocking
    from sd_agentic_rag.model_selector import ModelSelector
    selector = ModelSelector()

    # Step 1: Provider-specific override
    override = selector.get_embedding_dimensions_override()
    if override is not None:
        return override

    # Step 2: Known model map based on active model from model_selector
    model_name = selector.get_embedding_model()
    known_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "nomic-embed-text": 768,
        "nomic-embed-text-v1": 768,
        "qwen3-embedding": 4096,
        "qwen3-embedding:latest": 4096,
    }
    
    # Match the full name first; strip only a ":latest" tag. Other tags
    # (e.g. qwen3-embedding:0.6b) can have different sizes, so they are probed.
    if model_name in known_dimensions:
        return known_dimensions[model_name]
    if model_name.endswith(":latest"):
        base_model = model_name[:-len(":latest")]
        if base_model in known_dimensions:
            return known_dimensions[base_model]
    
    # Fallback: probe by embedding a short string
    return _dimension_probe(model_name)


def _dimension_probe(model_name: str) -> int:
    """One-time probe: embed a short string and return vector length.
    
    Called only when model is unknown and no override is configured.
    Caches result to avoid repeated probes. Calls the embeddings API directly
    (not EmbeddingService, whose constructor resolves dimensions itself).
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Vector length from probe embedding
    """
    # Module-level cache to avoid repeated probes
    if not hasattr(_dimension_probe, "_cache"):
        _dimension_probe._cache = {}
    
    if model_name in _dimension_probe._cache:
        return _dimension_probe._cache[model_name]
    
    try:
        import openai
        from sd_agentic_rag.model_selector import get_client_kwargs
        client = openai.OpenAI(**get_client_kwargs())
        response = client.embeddings.create(
            input=["probe"], model=model_name, encoding_format="float"
        )
        dims = len(response.data[0].embedding)
        _dimension_probe._cache[model_name] = dims
        return dims
    except Exception as e:
        raise RuntimeError(
            f"Failed to probe embedding dimensions for model '{model_name}'. "
            f"Ensure the model is running and accessible. Error: {e}"
        )



class StorePipeline:
    """Pipeline to process and store embeddings in Qdrant.

    Integrates Stage 2 (Embed) with Stage 3 (Store).
    """

    def __init__(self, vector_store: Optional[VectorStore] = None, skip_init: bool = False):
        """Initialize the storage pipeline.

        Args:
            vector_store: Optional VectorStore instance (creates default if not provided)
            skip_init: If True, skip VectorStore initialization (for testing)
        """
        self.vector_store = vector_store or VectorStore(skip_init=skip_init)

    def store_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 100,
        id_prefix: str = "chunk"
    ) -> Dict:
        """Process and store embedded chunks in Qdrant.

        Args:
            chunks: List of chunks with 'embedding' field
            batch_size: Number of points to store per batch
            id_prefix: Prefix for auto-generated IDs

        Returns:
            Dict with storage statistics
        """
        if not chunks:
            return {"stored": 0, "failed": 0, "skipped": 0}

        # Extract embedding dimension to validate
        sample_embedding = chunks[0].get("embedding", [])
        if sample_embedding:
            actual_dims = len(sample_embedding)
            expected_dims = self.vector_store.dimensions
            if actual_dims != expected_dims:
                print(f"Warning: Embedding dimensions mismatch. "
                      f"Expected {expected_dims}, got {actual_dims}")

        # Build points for upsert
        points = []
        for i, chunk in enumerate(chunks):
            point_id = chunk.get("chunk_id") or chunk.get("id")
            if not point_id:
                # Generate a proper UUID v4 for Qdrant
                point_id = str(uuid.uuid4())
            
            metadata = chunk.get("metadata", {})
            point = {
                "id": point_id,
                "vector": chunk.get("embedding", []),
                "payload": {
                    "ticket_id": chunk.get("ticket_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_text": chunk.get("chunk_text", ""),
                    # Ingest's metadata allow-list, so payload and metadata cannot drift
                    **{
                        field: metadata.get(field, metadata_default(field))
                        for field in METADATA_DEFAULTS
                    }
                }
            }
            points.append(point)

        # Upsert to Qdrant with error handling
        try:
            self.vector_store.upsert_points(points, batch_size=batch_size)
            stored_count = len(points)
        except Exception as e:
            # Handle dimension mismatch or other Qdrant errors
            if "dimension" in str(e).lower() or "dim" in str(e).lower():
                print(f"Dimension mismatch error: {e}")
                return {"stored": 0, "failed": len(points), "skipped": 0}
            raise

        return {
            "stored": stored_count,
            "failed": 0,
            "skipped": 0,
            "collection": self.vector_store.collection_name,
            "total_points": self.vector_store.count_points()
        }


def create_vector_store(**kwargs) -> VectorStore:
    """Factory function to create a VectorStore instance."""
    return VectorStore(**kwargs)


def create_store_pipeline(**kwargs) -> StorePipeline:
    """Factory function to create a StorePipeline instance.

    Args:
        **kwargs: Arguments for VectorStore creation or pre-created vector_store

    Returns:
        StorePipeline instance
    """
    skip_init = kwargs.get("skip_init", False)
    
    # If vector_store is provided, use it; otherwise create a new one
    if "vector_store" in kwargs:
        vector_store = kwargs["vector_store"]
    else:
        vector_store = VectorStore(**{k: v for k, v in kwargs.items() if k != "skip_init"})
    
    return StorePipeline(vector_store=vector_store, skip_init=skip_init)
