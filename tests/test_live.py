"""Live integration tests against Azure OpenAI gpt-5.4.

Run with:  pytest -m live -v
Skip with: pytest -m "not live"

These tests make real API calls and require `az login` authentication.
Each test generates a synthetic prompt with a known defect and verifies
the optimizer correctly identifies and fixes it.
"""

import pytest

from gpt_optimizer.optimizer import optimize_prompt
from gpt_optimizer.models import ChatMessage, OptimizerMode, PresetCheck, Role


pytestmark = pytest.mark.live


# ── Helpers ──────────────────────────────────────────────────────────


def _comments_text(response) -> str:
    """Join all comment reasons into a single searchable string."""
    return " ".join(c.reason.lower() for c in response.comments)


def _has_finding(response) -> bool:
    """Check if response contains any finding-type comments."""
    return any(c.kind == "finding" for c in response.comments)


# ── Test 1: Contradiction Detection ──────────────────────────────────


class TestContradictionDetection:
    """Prompt with contradictory instructions should be detected and resolved."""

    PROMPT = (
        "You are a data extraction assistant. "
        "Always return results as valid JSON objects. "
        "Do not use any structured or formatted output — respond in plain conversational text only. "
        "Include all extracted fields in your response."
    )

    def test_detects_contradiction(self):
        result = optimize_prompt(self.PROMPT)

        assert _has_finding(result), (
            f"Expected contradiction to be detected. "
            f"Comments: {[c.reason for c in result.comments]}"
        )
        text = _comments_text(result)
        assert any(word in text for word in ["contradict", "conflict", "json", "plain"]), (
            f"Expected comments to mention the JSON vs plain text conflict. Got: {text}"
        )

    def test_resolves_contradiction(self):
        result = optimize_prompt(self.PROMPT)

        new = result.new_developer_message.lower()
        # Should not contain BOTH conflicting clauses
        has_json = "json" in new
        has_plain_only = "plain conversational text only" in new
        assert not (has_json and has_plain_only), (
            f"Expected contradiction to be resolved, but both clauses remain: {result.new_developer_message}"
        )


# ── Test 2: Clean Prompt Passthrough ─────────────────────────────────


class TestCleanPromptPassthrough:
    """A well-structured prompt should pass through with minimal changes."""

    PROMPT = (
        "You are a Python coding assistant. Help users write clean, "
        "efficient Python code. When showing code, use markdown code blocks "
        "with syntax highlighting. Always explain your reasoning before "
        "showing code. Keep explanations concise."
    )

    def test_no_issues_found(self):
        result = optimize_prompt(self.PROMPT)

        # Should not have any finding-type comments (structural issues)
        findings = [c for c in result.comments if c.kind == "finding"]
        assert len(findings) == 0, (
            f"Expected no findings for a clean prompt. Got: {[f.reason for f in findings]}"
        )

    def test_preserves_intent(self):
        result = optimize_prompt(self.PROMPT)

        new = result.new_developer_message.lower()
        # Core intent keywords should survive
        for keyword in ["python", "code", "explain"]:
            assert keyword in new, (
                f"Expected '{keyword}' to be preserved. Got: {result.new_developer_message}"
            )


# ── Test 3: Format Ambiguity ─────────────────────────────────────────


class TestFormatAmbiguity:
    """Prompt asking for 'structured format' without specifying schema."""

    PROMPT = (
        "You are a document analysis assistant. Extract key information "
        "from uploaded documents including dates, names, monetary amounts, "
        "and action items. Return results in a structured format."
    )

    def test_flags_ambiguous_format(self):
        result = optimize_prompt(self.PROMPT)

        text = _comments_text(result)
        assert any(word in text for word in ["format", "schema", "structure", "field", "ambig"]), (
            f"Expected format ambiguity to be flagged. Comments: {text}"
        )

    def test_adds_format_spec(self):
        result = optimize_prompt(self.PROMPT)

        new = result.new_developer_message.lower()
        # The rewriter should add format clarity — look for schema indicators
        format_indicators = ["json", "field", "key", "format", "output", "schema", "```"]
        has_format = any(ind in new for ind in format_indicators)
        assert has_format, (
            f"Expected optimized prompt to include format specification. Got: {result.new_developer_message}"
        )


# ── Test 4: Custom Changes ──────────────────────────────────────────


class TestCustomChanges:
    """Custom change request should be applied while preserving tone."""

    PROMPT = (
        "You are a customer support agent. Help users with their questions. "
        "Be professional and empathetic. Always acknowledge the customer's "
        "concern before providing a solution."
    )
    CHANGES = "specialize this prompt for handling billing disputes and refund requests"

    def test_applies_customization(self):
        result = optimize_prompt(self.PROMPT, requested_changes=self.CHANGES)

        new = result.new_developer_message.lower()
        # Should mention billing/refunds/disputes
        assert any(word in new for word in ["billing", "refund", "dispute"]), (
            f"Expected billing/refund specialization. Got: {result.new_developer_message}"
        )

    def test_preserves_tone(self):
        result = optimize_prompt(self.PROMPT, requested_changes=self.CHANGES)

        new = result.new_developer_message.lower()
        # Should preserve professional/empathetic tone markers
        assert any(word in new for word in ["professional", "empathetic", "empathy", "acknowledge"]), (
            f"Expected professional/empathetic tone to be preserved. Got: {result.new_developer_message}"
        )


# ── Test 5: Preset Conflict Check ────────────────────────────────────


class TestPresetConflictCheck:
    """Targeted conflict check should detect word count contradiction."""

    PROMPT = (
        "You are a research summarizer. Keep all answers under 50 words. "
        "Provide thorough, detailed explanations of at least 500 words "
        "for every topic the user asks about."
    )

    def test_detects_conflict(self):
        result = optimize_prompt(
            self.PROMPT,
            preset_check=PresetCheck.conflicting_instructions,
        )

        assert result.issues_found is True, (
            f"Expected issues_found=True. Got: {result.issues_found}"
        )
        assert result.operation_mode == "preset_check"
        assert result.preset_check == "conflicting_instructions"

    def test_explains_conflict(self):
        result = optimize_prompt(
            self.PROMPT,
            preset_check=PresetCheck.conflicting_instructions,
        )

        text = _comments_text(result)
        assert any(word in text for word in ["50", "500", "word", "length", "conflict", "contradict"]), (
            f"Expected word count conflict in comments. Got: {text}"
        )


# ── Test 6: Few-Shot Inconsistency ───────────────────────────────────


class TestFewShotInconsistency:
    """Few-shot examples that violate the system prompt rules."""

    PROMPT = "Always respond with exactly one sentence. No exceptions."

    MESSAGES = [
        ChatMessage(role=Role.user, content="What is machine learning?"),
        ChatMessage(
            role=Role.assistant,
            content=(
                "Machine learning is a branch of artificial intelligence. "
                "It focuses on building systems that learn from data. "
                "These systems improve their performance over time without "
                "being explicitly programmed for every scenario."
            ),
        ),
    ]

    def test_detects_inconsistency(self):
        result = optimize_prompt(self.PROMPT, messages=self.MESSAGES)

        text = _comments_text(result)
        assert any(word in text for word in ["sentence", "few-shot", "example", "inconsist", "comply", "violat"]), (
            f"Expected few-shot inconsistency to be flagged. Comments: {text}"
        )

    def test_response_has_changes(self):
        result = optimize_prompt(self.PROMPT, messages=self.MESSAGES)

        # Either the prompt or the messages should have been modified
        has_prompt_change = result.new_developer_message != self.PROMPT
        has_message_change = len(result.new_messages) > 0 and any(
            (m.get("content") if isinstance(m, dict) else m.content) != msg.content
            for m, msg in zip(result.new_messages, self.MESSAGES)
        )
        assert has_prompt_change or has_message_change, (
            "Expected either prompt or messages to be modified"
        )
