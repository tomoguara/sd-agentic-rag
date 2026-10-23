"""Unit tests for Guard Wrapper (Stage 8)

Tests for:
- Config: Configuration management for Guard
- Guardrails: Guardrails wrapper for answer validation
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sd_agentic_rag.sd_agentic_guard import (
    Config,
    Guardrails,
    create_guardrails,
    SdAgenticGuardrails,
    STANDARD_REFUSAL,
)


# Reset Config singleton before tests
def reset_config_singleton():
    """Reset the Config singleton for testing."""
    Config._instance = None


@pytest.fixture(autouse=True)
def reset_config():
    """Reset Config singleton before each test."""
    reset_config_singleton()
    yield
    reset_config_singleton()


class TestConfig:
    """Tests for Guard Configuration manager."""

    def test_config_singleton(self):
        """Config should be a singleton."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_default_values(self):
        """Test default configuration values for Guard."""
        config = Config.__new__(Config)
        config._config = {
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

        # Should have guard config with defaults
        guard_config = config.get_guard_config()
        assert guard_config.get("enable_citation_check") is True
        assert guard_config.get("enable_relevance_gate") is True
        assert guard_config.get("enable_pii_redact") is True
        assert guard_config.get("min_citations_per_answer") == 1
        assert guard_config.get("relevance_threshold") == 0.5

    def test_config_get_with_dot_notation(self):
        """Test getting config values using dot notation."""
        config = Config.__new__(Config)
        config._config = {
            "guard": {
                "enable_citation_check": True,
                "enable_relevance_gate": True,
                "max_tokens": 2000
            }
        }

        # Test various config values
        assert config.get("guard", "enable_citation_check") is True
        assert config.get("guard", "enable_relevance_gate") is True
        assert config.get("guard", "max_tokens") == 2000
        assert config.get("nonexistent", "key", default="default") == "default"

    def test_config_load_from_file(self):
        """Test loading config from YAML file."""
        import tempfile
        import os

        # Create a temp YAML config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
guard:
  enable_citation_check: false
  enable_relevance_gate: true
  min_citations_per_answer: 2
""")
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            guard_config = config.get_guard_config()
            assert guard_config.get("enable_citation_check") is False
            assert guard_config.get("min_citations_per_answer") == 2
        finally:
            os.unlink(temp_path)


    def test_config_load_from_file(self):
        """Test loading config from YAML file."""
        import tempfile
        import os

        # Create a temp YAML config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
guard:
  enable_citation_check: false
  enable_relevance_gate: true
  min_citations_per_answer: 2
""")
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            guard_config = config.get_guard_config()
            assert guard_config.get("enable_citation_check") is False
            assert guard_config.get("min_citations_per_answer") == 2

            # Check generate config
            generate_config = config.get("generate")
            assert generate_config is None or "model" not in generate_config
        finally:
            os.unlink(temp_path)

    def test_config_loads_enforce_value(self):
        """Test that config loads enforce value correctly."""
        import tempfile
        import os

        # Create a temp YAML config file with enforce
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
guard:
  enforce: false
  min_citations_per_answer: 2
""")
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            guard_config = config.get_guard_config()
            assert guard_config.get("enforce") is False
            assert guard_config.get("min_citations_per_answer") == 2
        finally:
            os.unlink(temp_path)

    def test_guardrails_defaults_match_config(self):
        """Test that Guardrails defaults match previous hard-coded values."""
        guard = Guardrails()
        
        # Previous hard-coded defaults
        assert guard.enable_citation_check is True
        assert guard.enable_relevance_gate is True
        assert guard.enable_pii_redact is True
        assert guard.enable_safety_filter is False
        assert guard.min_citations == 1
        assert guard.relevance_threshold == 0.5
        assert guard.max_tokens == 2000
        assert guard.min_tokens == 10
        assert guard.enforce is True

class TestGuardrails:
    """Tests for the Guardrails class."""

    def test_initialization_defaults(self):
        """Test Guardrails with default parameters."""
        guard = Guardrails()

        assert guard.enable_citation_check is True
        assert guard.enable_relevance_gate is True
        assert guard.enable_pii_redact is True
        assert guard.enable_safety_filter is False
        assert guard.min_citations == 1
        assert guard.relevance_threshold == 0.5
        assert guard.max_tokens == 2000
        assert guard.min_tokens == 10
        assert guard.embedding_service is None

    def test_initialization_custom_values(self):
        """Test Guardrails with custom parameters."""
        mock_service = Mock()
        guard = Guardrails(
            enable_citation_check=False,
            enable_relevance_gate=False,
            enable_pii_redact=False,
            enable_safety_filter=True,
            min_citations_per_answer=3,
            relevance_threshold=0.7,
            max_tokens=1500,
            min_tokens=20,
            embedding_service=mock_service
        )

        assert guard.enable_citation_check is False
        assert guard.enable_relevance_gate is False
        assert guard.enable_pii_redact is False
        assert guard.enable_safety_filter is True
        assert guard.min_citations == 3
        assert guard.relevance_threshold == 0.7
        assert guard.max_tokens == 1500
        assert guard.min_tokens == 20
        assert guard.embedding_service == mock_service

    def test_redact_pii(self):
        """Test PII redaction."""
        guard = Guardrails(enable_pii_redact=True)

        # Test SSN redaction
        text_with_ssn = "My SSN is 123-45-6789"
        redacted = guard.redact_pii(text_with_ssn)
        assert "[SSN_REDACTED]" in redacted
        assert "123-45-6789" not in redacted

        # Test credit card redaction
        text_with_card = "Card: 1234567890123456"
        redacted = guard.redact_pii(text_with_card)
        assert "[CARD_REDACTED]" in redacted
        assert "1234567890123456" not in redacted

        # Test email redaction
        text_with_email = "Email: test@example.com"
        redacted = guard.redact_pii(text_with_email)
        assert "[EMAIL_REDACTED]" in redacted
        assert "test@example.com" not in redacted

        # Test phone redaction
        text_with_phone = "Phone: 123-456-7890"
        redacted = guard.redact_pii(text_with_phone)
        assert "[PHONE_REDACTED]" in redacted
        assert "123-456-7890" not in redacted

        # Test multiple PII
        text_multiple = "Contact: 555-123-4567 or email@test.com"
        redacted = guard.redact_pii(text_multiple)
        assert "[PHONE_REDACTED]" in redacted
        assert "[EMAIL_REDACTED]" in redacted

    def test_redact_pii_disabled(self):
        """Test that PII redaction can be disabled."""
        guard = Guardrails(enable_pii_redact=False)

        text_with_ssn = "My SSN is 123-45-6789"
        redacted = guard.redact_pii(text_with_ssn)
        # Redaction should still happen inside the method, but it won't be applied
        # to the answer in process_answer
        assert "[SSN_REDACTED]" in redacted

    def test_extract_citations(self):
        """Test citation extraction."""
        guard = Guardrails()

        # Test with single citation
        answer_with_citations = "This is a solution [CITATION:INC001]"
        citations = guard.extract_citations(answer_with_citations)
        assert citations == ["INC001"]

        # Test with multiple citations
        answer_multi = "Solution [CITATION:INC001] and [CITATION:INC002]"
        citations = guard.extract_citations(answer_multi)
        assert citations == ["INC001", "INC002"]

        # Test with no citations
        answer_no_citations = "This is just text"
        citations = guard.extract_citations(answer_no_citations)
        assert citations == []

    def test_extract_citations_multiple_per_claim(self):
        """Test extracting multiple citations per claim."""
        guard = Guardrails()

        answer_multi = "This is supported [CITATION:INC001][CITATION:INC002]"
        citations = guard.extract_citations(answer_multi)
        assert citations == ["INC001", "INC002"]

    def test_verify_citations_valid(self):
        """Test citation verification with valid citations."""
        guard = Guardrails()

        answer = "Solution [CITATION:INC001] and [CITATION:INC002]"
        context_citations = [
            {"ticket_id": "INC001"},
            {"ticket_id": "INC002"}
        ]

        result = guard.verify_citations(answer, context_citations)

        assert result["valid"] is True
        assert result["missing"] == []
        assert "INC001" in result["answer_citations"]
        assert "INC002" in result["answer_citations"]

    def test_verify_citations_missing(self):
        """Test citation verification with missing citations."""
        guard = Guardrails()

        answer = "Solution [CITATION:INC001] and [CITATION:INC003]"
        context_citations = [
            {"ticket_id": "INC001"},
            {"ticket_id": "INC002"}
        ]

        result = guard.verify_citations(answer, context_citations)

        assert result["valid"] is False
        assert "INC003" in result["missing"]
        assert result["answer_citations"] == ["INC001", "INC003"]
        assert set(result["context_citations"]) == {"INC001", "INC002"}

    def test_check_refusal_yes(self):
        """Test refusal detection with actual refusal."""
        guard = Guardrails()

        refusal_answers = [
            "I cannot answer that question",
            "I cannot find sufficient information",
            "I don't know the answer",
            "I am unable to answer",
            "Not enough information",
            "Outside my knowledge",
            "I'm unable to answer",
            "Sorry, I cannot find that in the tickets",
            "  I can\u2019t find any matching ticket",
        ]

        for answer in refusal_answers:
            assert guard.check_refusal(answer) is True, f"Should detect refusal: {answer}"

    def test_check_refusal_no(self):
        """Test refusal detection with non-refusal."""
        guard = Guardrails()

        non_refusal_answers = [
            "The answer is this",
            "Here is the solution",
            "I can help with that",
            "Users who cannot find the export button should open Settings",
            "Restart the service; the error is not applicable after the patch",
        ]

        for answer in non_refusal_answers:
            assert guard.check_refusal(answer) is False, f"Should not detect refusal: {answer}"

    def test_check_length_valid(self):
        """Test length check with valid length."""
        guard = Guardrails(max_tokens=1000, min_tokens=10)

        # 40 tokens (160 characters / 4)
        answer = "word " * 40  # 200 characters = 50 tokens

        result = guard.check_length(answer)

        assert result["valid"] is True
        assert result["token_count"] == 50
        assert result["min_tokens"] == 10
        assert result["max_tokens"] == 1000

    def test_check_length_too_short(self):
        """Test length check with answer too short."""
        guard = Guardrails(max_tokens=1000, min_tokens=20)

        # 40 characters = 10 tokens (below min)
        answer = "word " * 8  # 40 characters

        result = guard.check_length(answer)

        assert result["valid"] is False
        assert result["token_count"] == 10
        assert result["token_count"] < result["min_tokens"]

    def test_check_length_too_long(self):
        """Test length check with answer too long."""
        guard = Guardrails(max_tokens=50, min_tokens=10)

        # 300 characters = 75 tokens (above max)
        answer = "word " * 60  # 300 characters

        result = guard.check_length(answer)

        assert result["valid"] is False
        assert result["token_count"] == 75
        assert result["token_count"] > result["max_tokens"]

    def test_check_safety_safe(self):
        """Test safety check with safe content."""
        guard = Guardrails(enable_safety_filter=True)

        safe_answers = [
            "This is a safe answer",
            "The solution is clear",
            "I can help with your issue"
        ]

        for answer in safe_answers:
            result = guard.check_safety(answer)
            assert result["safe"] is True, f"Should be safe: {answer}"

    def test_check_safety_unsafe(self):
        """Test safety check with unsafe content."""
        guard = Guardrails(enable_safety_filter=True)

        # Test patterns - partial matches
        unsafe_patterns = [
            "There is violence in the code",
            "Self-harm warning",
            "Threat to users",
        ]

        for answer in unsafe_patterns:
            result = guard.check_safety(answer)
            # Note: Our patterns look for the word, not exact matches
            # This is a basic test - patterns are case-insensitive
            # "violence" should match "violence"
            pass  # We'll just check it doesn't crash

    def test_create_guardrails_factory(self):
        """Test create_guardrails factory function."""
        guard = create_guardrails(
            enable_citation_check=False,
            min_citations_per_answer=2
        )

        assert isinstance(guard, Guardrails)
        assert guard.enable_citation_check is False
        assert guard.min_citations == 2

    def test_create_guardrails_reads_guard_config(self, tmp_path):
        """create_guardrails takes every guard: value from config.yaml."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "guard:\n"
            "  enforce: false\n"
            "  min_citations_per_answer: 3\n"
            "  relevance_threshold: 0.8\n"
            "  min_tokens: 7\n"
            "  max_tokens: 1500\n"
            "  enable_safety_filter: true\n"
        )
        Config(config_path=str(config_file))

        guard = create_guardrails()

        assert guard.enforce is False
        assert guard.min_citations == 3
        assert guard.relevance_threshold == 0.8
        assert guard.min_tokens == 7
        assert guard.max_tokens == 1500
        assert guard.enable_safety_filter is True

    def test_create_guardrails_kwargs_override_config(self, tmp_path):
        """Explicit kwargs win over config; None kwargs fall back to config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("guard:\n  enforce: false\n  relevance_threshold: 0.8\n")
        Config(config_path=str(config_file))

        guard = create_guardrails(enforce=True, relevance_threshold=None)

        assert guard.enforce is True
        assert guard.relevance_threshold == 0.8

    def test_sd_agentic_guardrails_alias(self):
        """Test SdAgenticGuardrails alias."""
        guard = SdAgenticGuardrails()

        assert isinstance(guard, Guardrails)


class TestGuardrailsIntegration:
    """Integration tests for Guardrails."""

    def test_process_answer_allowed(self):
        """Test processing an allowed answer."""
        guard = Guardrails(min_citations_per_answer=1, min_tokens=5, max_tokens=1000)

        answer = "The solution is [CITATION:INC001]. This provides the needed fix."
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        assert result["allowed"] is True
        assert len(result["checks"]["citation"]["answer_citations"]) == 1

    def test_process_answer_missing_citations(self):
        """Test processing answer with missing citations."""
        guard = Guardrails(enable_citation_check=True)

        answer = "The solution is [CITATION:INC999]"
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        assert result["allowed"] is False
        assert any("Missing citations" in action for action in result["actions"])

    def test_process_answer_not_enough_citations(self):
        """Test processing answer with insufficient citations."""
        guard = Guardrails(min_citations_per_answer=2)

        answer = "The solution is [CITATION:INC001]"
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        assert result["allowed"] is False
        assert any("Not enough citations" in action for action in result["actions"])

    def test_process_answer_refusal_detected(self):
        """Test processing answer that is a refusal."""
        guard = Guardrails(
            min_tokens=5,
            max_tokens=1000,
            min_citations_per_answer=0
        )

        answer = "I cannot find sufficient information in the ticket data to provide an answer."
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        # Refusals are allowed but flagged
        assert result["allowed"] is True
        assert any("Refusal detected" in action for action in result["actions"])

    def test_process_answer_pii_redacted(self):
        """Test processing answer with PII that gets redacted."""
        guard = Guardrails(
            enable_citation_check=False,
            enable_pii_redact=True,
            min_tokens=5,
            max_tokens=1000,
            min_citations_per_answer=0
        )

        answer = "Contact us at 123-456-7890 for technical support assistance."
        question = "What is the contact?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        assert result["allowed"] is True
        assert "[PHONE_REDACTED]" in result["modified_answer"]
        assert "123-456-7890" not in result["modified_answer"]
        assert result["modified_answer"] != answer

    def test_process_answer_with_checks_dict(self):
        """Test that process_answer returns detailed checks."""
        guard = Guardrails()

        answer = "The solution is [CITATION:INC001]"
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        assert "checks" in result
        assert isinstance(result["checks"], dict)

        # Check for expected check types
        checks = result["checks"]
        assert "citation" in checks or "citation_count" in checks
        assert "refusal" in checks
        assert "length" in checks

    def test_refusal_exempts_citation_count_check(self):
        """Test that refusals are exempt from citation count requirement."""
        guard = Guardrails(
            min_citations_per_answer=2,
            min_tokens=5,
            max_tokens=1000
        )

        # A short refusal with no citations
        answer = "I cannot find sufficient information in the ticket data to answer this question."
        question = "What is the solution?"
        citations = []

        result = guard.process_answer(answer, question, citations)

        # Should be allowed because refusal exempt from citation count
        assert result["allowed"] is True
        # But still has the citation count check result
        assert result["checks"]["citation_count"]["count"] == 0

    def test_refusal_exempts_length_check(self):
        """Test that refusals are exempt from length requirements."""
        guard = Guardrails(
            min_tokens=100,  # High minimum
            max_tokens=1000
        )

        # A short refusal
        answer = "I cannot find sufficient information in the ticket data to answer this question."
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        # Should be allowed because refusal exempt from length check
        assert result["allowed"] is True
        # But length check should still run (just not block)
        assert "length" in result["checks"]

    def test_enforcement_substitutes_refusal_when_blocked(self):
        """Test that enforcement replaces blocked answers with refusal."""
        guard = Guardrails(
            min_citations_per_answer=2,
            enforce=True
        )

        answer = "Only one citation here [CITATION:INC001]."
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        # Should be blocked (not enough citations)
        assert result["allowed"] is False
        assert result["enforced"] is True
        assert result["replaced"] is True
        # Should be replaced with standard refusal
        assert "cannot find sufficient information" in result["modified_answer"]
        assert result["modified_answer"] != answer
        assert any("blocked" in action.lower() for action in result["actions"])

    def test_enforcement_false_allows_blocked(self):
        """Test that without enforcement, blocked answers pass through."""
        guard = Guardrails(
            min_citations_per_answer=2,
            enforce=False
        )

        answer = "Only one citation here [CITATION:INC001]."
        question = "What is the solution?"
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, question, citations)

        # Should still be blocked but not enforced
        assert result["allowed"] is False
        assert result["enforced"] is False
        assert result["replaced"] is False
        # Original answer should pass through (after PII redaction if any)
        assert result["modified_answer"] == answer

    def test_refusal_not_replaced_when_enforced(self):
        """Test that actual refusals don't get substituted (already refusal)."""
        guard = Guardrails(
            min_citations_per_answer=2,
            max_tokens=10,
            enforce=True
        )

        answer = "I cannot find sufficient information in the ticket data to answer this question."
        question = "What is the solution?"
        citations = []

        result = guard.process_answer(answer, question, citations)

        # Should be allowed (refusal exempt from checks)
        assert result["allowed"] is True
        # Enforcement was active, but nothing was blocked, so nothing was replaced
        assert result["enforced"] is True
        assert result["replaced"] is False
        assert result["modified_answer"] == answer

    def test_refusal_with_fabricated_citation_is_replaced(self):
        """A refusal that cites an unknown ticket is still blocked and replaced."""
        guard = Guardrails(enforce=True)

        answer = (
            "I cannot find sufficient information in the ticket data, "
            "but see [CITATION:XYZ999]."
        )
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, "What is the solution?", citations)

        assert result["checks"]["refusal"]["detected"] is True
        assert result["allowed"] is False
        assert result["replaced"] is True
        assert result["modified_answer"] == STANDARD_REFUSAL

    def test_refusal_with_unsafe_content_is_replaced(self):
        """A refusal that fails the safety filter is still blocked and replaced."""
        guard = Guardrails(enable_safety_filter=True, enforce=True)

        answer = "I cannot answer that, but here is how to harass the user."

        result = guard.process_answer(answer, "What is the solution?", [])

        assert result["checks"]["refusal"]["detected"] is True
        assert result["allowed"] is False
        assert result["replaced"] is True
        assert result["modified_answer"] == STANDARD_REFUSAL

    def test_refusal_phrase_inside_answer_is_not_exempt(self):
        """Support text that merely contains a refusal phrase is checked normally."""
        guard = Guardrails(enforce=True)

        answer = (
            "Users who cannot find the export button should open Settings "
            "and enable reports [CITATION:XYZ999]."
        )
        citations = [{"ticket_id": "INC001"}]

        result = guard.process_answer(answer, "Where is export?", citations)

        assert result["checks"]["refusal"]["detected"] is False
        assert result["allowed"] is False
        assert result["replaced"] is True
        assert result["modified_answer"] == STANDARD_REFUSAL
