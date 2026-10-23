"""Unit tests for Stage 3: Storage (VectorStore)"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sd_agentic_rag.sd_agentic_store import (
    Config,
    VectorStore,
    StorePipeline,
    create_vector_store,
    create_store_pipeline,
    get_embedding_model_dimensions
)


class TestConfig:
    """Tests for Configuration manager"""

    def test_config_singleton(self):
        """Config should be a singleton"""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_default_values(self):
        """Test default configuration values"""
        config = Config()

        # Should have qdrant config with defaults
        qdrant_config = config.get_qdrant_config()
        assert qdrant_config.get("host") == "localhost"
        assert qdrant_config.get("port") == 6333
        assert qdrant_config.get("collection_name") == "tickets"

    def test_config_get_with_dot_notation(self):
        """Test getting config values using dot notation"""
        config = Config()
        assert config.get("openai", "api_key_env") == "OPENAI_API_KEY"
        assert config.get("ingest", "chunk_size") == 1000


class TestGetEmbeddingModelDimensions:
    """Tests for get_embedding_model_dimensions: override -> known map -> probe"""

    @pytest.fixture(autouse=True)
    def no_dimension_overrides(self, monkeypatch):
        """sd-agentic-rag.cfg is loaded into os.environ at import; start each test clean"""
        monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL_DIMM", raising=False)
        monkeypatch.delenv("OPENAI_EMBEDDING_MODEL_DIMM", raising=False)
        if hasattr(get_embedding_model_dimensions.__globals__["_dimension_probe"], "_cache"):
            get_embedding_model_dimensions.__globals__["_dimension_probe"]._cache.clear()

    @pytest.mark.parametrize("model,expected", [
        ("text-embedding-3-small", 1536),
        ("text-embedding-3-large", 3072),
        ("text-embedding-ada-002", 1536),
        ("qwen3-embedding:latest", 4096),
        ("nomic-embed-text", 768),
    ])
    def test_known_model_map(self, reset_model_selector, model, expected):
        """Without an override, known models resolve from the map"""
        from sd_agentic_rag.model_selector import ModelSelector
        reset_model_selector("openai")
        with patch.object(ModelSelector, 'get_embedding_model', return_value=model):
            assert get_embedding_model_dimensions() == expected

    @pytest.mark.parametrize("use_ollama,var", [
        ("true", "OLLAMA_EMBEDDING_MODEL_DIMM"),
        ("false", "OPENAI_EMBEDDING_MODEL_DIMM"),
    ])
    def test_active_provider_override_wins_over_known_map(self, reset_model_selector, monkeypatch, use_ollama, var):
        """The active provider's *_DIMM override beats the known-model map"""
        from sd_agentic_rag.model_selector import ModelSelector
        reset_model_selector("ollama" if use_ollama == "true" else "openai")
        monkeypatch.setenv(var, "1024")
        with patch.object(ModelSelector, 'get_embedding_model', return_value="text-embedding-3-small"):
            assert get_embedding_model_dimensions() == 1024

    def test_other_providers_override_is_ignored(self, reset_model_selector, monkeypatch):
        """In OpenAI mode, OLLAMA_EMBEDDING_MODEL_DIMM has no effect"""
        from sd_agentic_rag.model_selector import ModelSelector
        reset_model_selector("openai")
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL_DIMM", "4096")
        with patch.object(ModelSelector, 'get_embedding_model', return_value="text-embedding-3-small"):
            assert get_embedding_model_dimensions() == 1536

    @pytest.mark.parametrize("value", ["abc", "0", "-5", "12.5"])
    def test_invalid_override_errors(self, reset_model_selector, monkeypatch, value):
        """A non-positive-integer override fails with a message naming the variable"""
        reset_model_selector("ollama")
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL_DIMM", value)
        with pytest.raises(ValueError, match="OLLAMA_EMBEDDING_MODEL_DIMM"):
            get_embedding_model_dimensions()

    def test_config_yaml_dimension_key_is_not_used(self, reset_model_selector):
        """openai.embedding_dimensions in config.yaml no longer affects dimensions"""
        from sd_agentic_rag.model_selector import ModelSelector
        reset_model_selector("openai")
        with patch('sd_agentic_rag.sd_agentic_store.Config') as MockConfig:
            MockConfig.return_value.get.return_value = 4096
            with patch.object(ModelSelector, 'get_embedding_model', return_value="text-embedding-3-small"):
                assert get_embedding_model_dimensions() == 1536

    def test_unknown_model_probes_once(self, reset_model_selector):
        """Unknown models are probed via the embeddings API (no recursion) and cached"""
        from sd_agentic_rag.model_selector import ModelSelector
        reset_model_selector("ollama")
        client = MagicMock()
        client.embeddings.create.return_value.data = [Mock(embedding=[0.0] * 1024)]
        with patch.object(ModelSelector, 'get_embedding_model', return_value="mxbai-embed-large"), \
             patch('openai.OpenAI', return_value=client):
            assert get_embedding_model_dimensions() == 1024
            assert get_embedding_model_dimensions() == 1024
        client.embeddings.create.assert_called_once()
        assert client.embeddings.create.call_args.kwargs["model"] == "mxbai-embed-large"

    @pytest.mark.parametrize("model", ["qwen3-embedding:0.6b", "qwen3-embedding:4b"])
    def test_tagged_model_is_probed_not_mapped(self, reset_model_selector, model):
        """Non-:latest tags are not mapped to the family's size; they are probed"""
        from sd_agentic_rag.model_selector import ModelSelector
        reset_model_selector("ollama")
        client = MagicMock()
        client.embeddings.create.return_value.data = [Mock(embedding=[0.0] * 1024)]
        with patch.object(ModelSelector, 'get_embedding_model', return_value=model), \
             patch('openai.OpenAI', return_value=client):
            assert get_embedding_model_dimensions() == 1024
        client.embeddings.create.assert_called_once()

    def test_latest_tag_maps_to_base_model(self, reset_model_selector):
        """A :latest tag resolves via the untagged entry"""
        from sd_agentic_rag.model_selector import ModelSelector
        reset_model_selector("ollama")
        with patch.object(ModelSelector, 'get_embedding_model', return_value="nomic-embed-text:latest"):
            assert get_embedding_model_dimensions() == 768


class TestStorePipeline:
    """Tests for StorePipeline class - the main integration point"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        mock_store = Mock(spec=VectorStore)
        mock_store.dimensions = 1536
        mock_store.upsert_points = Mock()
        mock_store.count_points = Mock(return_value=100)
        mock_store.collection_name = "test_collection"
        return mock_store

    def test_store_chunks_with_valid_data(self, mock_vector_store):
        """Test storing chunks with valid embeddings"""
        chunks = [
            {
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Test chunk 1",
                "embedding": [0.1] * 1536,
                "metadata": {
                    "service": "network",
                    "severity": "high",
                    "ticket_type": "Incident",
                    "created_at": "2026-01-15",
                    "user_name": "John",
                    "user_email": "john@example.com"
                }
            },
            {
                "ticket_id": "TICK-002",
                "chunk_index": 0,
                "chunk_text": "Test chunk 2",
                "embedding": [0.2] * 1536,
                "metadata": {
                    "service": "database",
                    "severity": "medium",
                    "ticket_type": "Service Request",
                    "created_at": "2026-01-16",
                    "user_name": "Jane",
                    "user_email": "jane@example.com"
                }
            }
        ]

        pipeline = StorePipeline(vector_store=mock_vector_store)
        result = pipeline.store_chunks(chunks, batch_size=10)

        assert result["stored"] == 2
        assert result["collection"] == "test_collection"
        assert result["total_points"] == 100

    def test_store_chunks_empty_list(self, mock_vector_store):
        """Test storing empty chunk list"""
        pipeline = StorePipeline(vector_store=mock_vector_store)
        result = pipeline.store_chunks([])

        assert result["stored"] == 0

    def test_store_chunks_dimension_mismatch_warning(self, mock_vector_store, capsys):
        """Test dimension mismatch warning"""
        chunks = [
            {
                "embedding": [0.1] * 100,  # Wrong dimension
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Test chunk",
                "metadata": {}
            }
        ]

        mock_vector_store.dimensions = 1536
        mock_vector_store.upsert_points = Mock()
        mock_vector_store.count_points = Mock(return_value=1)

        pipeline = StorePipeline(vector_store=mock_vector_store)
        pipeline.store_chunks(chunks)

        # Should print warning about dimension mismatch
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "mismatch" in captured.out.lower()


class TestFactoryFunctions:
    """Tests for factory functions"""

    def test_create_vector_store(self):
        """Test create_vector_store factory function"""
        with patch('sd_agentic_rag.sd_agentic_store.VectorStore') as MockStore:
            MockStore.return_value = Mock(spec=VectorStore)
            store = create_vector_store(host="localhost", port=6333)
            assert store is not None

    def test_create_store_pipeline(self):
        """Test create_store_pipeline factory function"""
        with patch('sd_agentic_rag.sd_agentic_store.VectorStore') as MockStore:
            MockStore.return_value = Mock(spec=VectorStore)
            pipeline = create_store_pipeline()
            assert pipeline is not None
            assert pipeline.vector_store is not None

class TestPayloadIndexes:
    """Tests for payload index creation (Task 3.5)"""

    def test_create_collection_creates_payload_indexes(self):
        """Test that _create_collection creates keyword indexes for required fields"""
        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client

            with patch('sd_agentic_rag.sd_agentic_store.Config') as MockConfig:
                MockConfig.return_value.get.return_value = None

                # Create VectorStore with skip_init=True to avoid dimension verification
                store = VectorStore(skip_init=True)
                store.collection_name = "test_collection"

                # Mock the client to simulate collection creation
                mock_client.collection_exists.return_value = False

                # Call _create_collection
                store._create_collection()

                # Verify payload indexes were created for each required field
                required_fields = ["ticket_id", "service", "severity", "ticket_type", "status"]
                assert mock_client.create_payload_index.call_count == 5
                
                # Check each field was indexed
                calls = mock_client.create_payload_index.call_args_list
                indexed_fields = [call[1]['field_name'] for call in calls]
                for field_name in required_fields:
                    assert field_name in indexed_fields

                # Verify all fields are KEYWORD type
                for call in calls:
                    assert call[1]['field_schema'] == 'keyword' or 'KEYWORD' in str(call[1]['field_schema'])


class TestWipe:
    """--wipe drops and recreates the collection with the current dimensions"""

    def _store(self, mock_client, dimensions=1536):
        with patch('qdrant_client.QdrantClient', return_value=mock_client), \
             patch('sd_agentic_rag.sd_agentic_store.Config') as MockConfig:
            MockConfig.return_value.get_qdrant_config.return_value = {}
            store = VectorStore(collection_name="tickets_test", dimensions=dimensions, skip_init=True)
        return store

    def test_wipe_recreates_collection_at_current_dimensions(self):
        """An existing 4096-d collection is dropped and recreated at 1536"""
        mock_client = MagicMock()
        mock_client.count.side_effect = [Mock(count=42), Mock(count=0)]
        store = self._store(mock_client, dimensions=1536)

        result = store.wipe()

        assert result["success"] is True
        assert result["before_count"] == 42
        assert result["after_count"] == 0
        mock_client.delete_collection.assert_called_once_with(collection_name="tickets_test")
        vectors_config = mock_client.create_collection.call_args.kwargs["vectors_config"]
        assert vectors_config.size == 1536

    def test_wipe_creates_missing_collection(self):
        """Wiping a collection that does not exist just creates it"""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("not found")
        mock_client.count.return_value = Mock(count=0)
        store = self._store(mock_client)

        result = store.wipe()

        assert result["success"] is True
        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_called_once()

    def test_ingest_wipe_skips_dimension_check(self, monkeypatch):
        """ingest --wipe builds the store with skip_init=True and wipes before storing"""
        import sd_agentic_rag
        from sd_agentic_rag.rag_commands.rag_ingest import RagIngestCommand

        store = MagicMock()
        store.wipe.return_value = {"success": True}
        factory = MagicMock(return_value=store)
        monkeypatch.setattr(sd_agentic_rag, "create_vector_store", factory)
        monkeypatch.setattr(sd_agentic_rag, "SdAgenticIngest", MagicMock())
        monkeypatch.setattr(sd_agentic_rag, "EmbeddingService", MagicMock())
        embed_pipeline = MagicMock()
        embed_pipeline.return_value.process_chunks.return_value = []
        monkeypatch.setattr(sd_agentic_rag, "EmbeddingPipeline", embed_pipeline)
        monkeypatch.setattr(sd_agentic_rag, "StorePipeline", MagicMock())

        result = RagIngestCommand(debug=True).run("data.json", wipe=True)

        factory.assert_called_once_with(skip_init=True)
        store.wipe.assert_called_once()
        assert "error" not in result


class TestPayloadAllowList:
    """The Qdrant payload is built from ingest's metadata allow-list (design D3)"""

    def test_payload_keys_and_missing_defaults(self):
        from sd_agentic_rag.sd_agentic_metadata import METADATA_DEFAULTS

        mock_store = Mock()
        pipeline = StorePipeline(vector_store=mock_store)
        chunk = {
            "ticket_id": "INC1",
            "chunk_index": 0,
            "chunk_text": "text",
            "embedding": [0.1, 0.2],
            "metadata": {"title": "VPN drops", "user_email": "jane@acme.com"},
        }

        pipeline.store_chunks([chunk])

        payload = mock_store.upsert_points.call_args.args[0][0]["payload"]
        assert set(payload) == {"ticket_id", "chunk_index", "chunk_text", *METADATA_DEFAULTS}
        assert payload["title"] == "VPN drops"
        assert payload["tags"] == []
        assert payload["sla_minutes"] is None
        assert "user_email" not in payload
