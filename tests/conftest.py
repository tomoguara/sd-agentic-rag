"""Shared fixtures for the unit and functional tests."""
import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def reset_model_selector():
    """Fixture that resets the model_selector singleton and can set USE_OLLAMA_MODELS"""
    import sd_agentic_rag.model_selector as ms

    # Store original state
    original_selector = getattr(ms, '_model_selector', None)
    original_env = os.environ.get('USE_OLLAMA_MODELS')

    # Reset the singleton
    ms._model_selector = None

    # Set the mode if provided
    def _reset(mode=None):
        ms._model_selector = None
        if mode is not None:
            if mode == 'ollama':
                os.environ['USE_OLLAMA_MODELS'] = 'true'
            elif mode == 'openai':
                os.environ['USE_OLLAMA_MODELS'] = 'false'
            else:
                if 'USE_OLLAMA_MODELS' in os.environ:
                    del os.environ['USE_OLLAMA_MODELS']
        else:
            if 'USE_OLLAMA_MODELS' in os.environ:
                del os.environ['USE_OLLAMA_MODELS']

        # Create a fresh selector
        ms.get_model_selector()

    yield _reset

    # Restore original state
    ms._model_selector = original_selector
    if original_env is not None:
        os.environ['USE_OLLAMA_MODELS'] = original_env
    elif 'USE_OLLAMA_MODELS' in os.environ:
        del os.environ['USE_OLLAMA_MODELS']
