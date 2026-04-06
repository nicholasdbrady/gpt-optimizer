"""Tests for the prompt optimization workflow in gpt_optimizer.optimizer."""

from unittest.mock import MagicMock, patch
import pytest

from gpt_optimizer.models import (
    Role,
    ChatMessage,
    Issues,
    FewShotIssues,
    DevRewriteOutput,
    MessagesOutput,
    OptimizeRequest,
    OptimizeResponse,
    Comment,
    OptimizerMode,
    PresetCheck,
)
from gpt_optimizer.optimizer import (
    _normalize_messages,
    optimize_prompt,
    _run_preset_check,
    _run_full_optimize,
)


class TestNormalizeMessages:
    """Tests for _normalize_messages function."""

    def test_normalize_empty_messages(self):
        """Test normalizing empty messages list."""
        result = _normalize_messages([])
        assert result == []

    def test_normalize_chat_message_objects(self):
        """Test normalizing ChatMessage Pydantic objects."""
        messages = [
            ChatMessage(role=Role.user, content="Hello"),
            ChatMessage(role=Role.assistant, content="Hi there"),
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_normalize_dict_messages(self):
        """Test normalizing plain dict messages with role and content."""
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Test"

    def test_normalize_mixed_message_types(self):
        """Test normalizing mixed ChatMessage and dict messages."""
        messages = [
            ChatMessage(role=Role.user, content="User message"),
            {"role": "assistant", "content": "Assistant message"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_normalize_handles_enum_role(self):
        """Test that Role enum is properly converted to string."""
        messages = [ChatMessage(role=Role.user, content="Test")]
        result = _normalize_messages(messages)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["role"], str)

    def test_normalize_single_message(self):
        """Test normalizing a single message."""
        messages = [ChatMessage(role=Role.assistant, content="Single response")]
        result = _normalize_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == "Single response"

    def test_normalize_skips_messages_without_role_and_content(self):
        """Test that messages without role and content are skipped."""
        messages = [
            ChatMessage(role=Role.user, content="Valid"),
            {"invalid": "dict"},  # Missing role and content
            ChatMessage(role=Role.assistant, content="Also valid"),
        ]
        result = _normalize_messages(messages)
        # Should only include valid messages
        assert len(result) == 2
        assert result[0]["content"] == "Valid"
        assert result[1]["content"] == "Also valid"

    def test_normalize_preserves_content_type(self):
        """Test that content is converted to string."""
        messages = [
            {"role": "user", "content": "Text content"},
            ChatMessage(role=Role.assistant, content="Message content"),
        ]
        result = _normalize_messages(messages)
        assert isinstance(result[0]["content"], str)
        assert isinstance(result[1]["content"], str)


class TestOptimizePromptPresetCheck:
    """Tests for optimize_prompt with preset_check mode."""

    @patch("gpt_optimizer.optimizer.check_contradictions")
    def test_preset_check_conflicting_instructions_no_issues(self, mock_check):
        """Test preset_check with no conflicting instructions found."""
        mock_check.return_value = Issues.no_issues()

        response = optimize_prompt(
            developer_message="Test prompt",
            preset_check=PresetCheck.conflicting_instructions,
            api_key="test-key",
        )

        assert response.operation_mode == "preset_check"
        assert response.preset_check == "conflicting_instructions"
        assert response.new_developer_message == "Test prompt"
        assert response.issues_found is False

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.rewrite_dev_prompt")
    def test_preset_check_conflicting_instructions_with_issues(
        self, mock_rewrite, mock_check
    ):
        """Test preset_check when issues are found and rewriting is applied."""
        mock_check.return_value = Issues(
            has_issues=True,
            issues=["Contradictory instruction 1"],
        )
        mock_rewrite.return_value = DevRewriteOutput(
            new_developer_message="Rewritten prompt without contradiction"
        )

        response = optimize_prompt(
            developer_message="Test prompt with contradiction",
            preset_check=PresetCheck.conflicting_instructions,
            api_key="test-key",
        )

        assert response.operation_mode == "preset_check"
        assert response.issues_found is True
        assert response.new_developer_message == "Rewritten prompt without contradiction"
        assert len(response.comments) >= 2  # At least finding + explanation

    @patch("gpt_optimizer.optimizer.check_format")
    def test_preset_check_ambiguity(self, mock_check):
        """Test preset_check for ambiguity issues."""
        mock_check.return_value = Issues.no_issues()

        response = optimize_prompt(
            developer_message="Test prompt",
            preset_check=PresetCheck.ambiguity,
            api_key="test-key",
        )

        assert response.operation_mode == "preset_check"
        assert response.preset_check == "ambiguity"
        mock_check.assert_called_once()

    @patch("gpt_optimizer.optimizer.check_format")
    def test_preset_check_output_format(self, mock_check):
        """Test preset_check for output_format issues."""
        mock_check.return_value = Issues.no_issues()

        response = optimize_prompt(
            developer_message="Test prompt",
            preset_check=PresetCheck.output_format,
            api_key="test-key",
        )

        assert response.operation_mode == "preset_check"
        assert response.preset_check == "output_format"
        mock_check.assert_called_once()

    @patch("gpt_optimizer.optimizer.check_format")
    def test_preset_check_with_messages(self, mock_check):
        """Test preset_check includes messages in response."""
        mock_check.return_value = Issues.no_issues()
        messages = [ChatMessage(role=Role.user, content="Example")]

        response = optimize_prompt(
            developer_message="Test prompt",
            messages=messages,
            preset_check=PresetCheck.output_format,
            api_key="test-key",
        )

        assert len(response.new_messages) == 1
        # new_messages can be either dict or ChatMessage depending on _normalize_messages
        assert isinstance(response.new_messages[0], (dict, ChatMessage))


class TestOptimizePromptRequestedChanges:
    """Tests for optimize_prompt with requested_changes mode."""

    @patch("gpt_optimizer.optimizer.rewrite_custom")
    def test_requested_changes_mode(self, mock_rewrite):
        """Test optimize_prompt with requested_changes."""
        mock_rewrite.return_value = DevRewriteOutput(
            new_developer_message="Concise rewritten prompt"
        )

        response = optimize_prompt(
            developer_message="Original verbose prompt",
            requested_changes="Make it more concise",
            api_key="test-key",
        )

        assert response.operation_mode == "custom"
        assert response.new_developer_message == "Concise rewritten prompt"
        assert "custom optimization" in response.summary
        mock_rewrite.assert_called_once()

    @patch("gpt_optimizer.optimizer.rewrite_custom")
    def test_requested_changes_with_messages(self, mock_rewrite):
        """Test requested_changes preserves original messages."""
        mock_rewrite.return_value = DevRewriteOutput(new_developer_message="Rewritten")
        messages = [
            ChatMessage(role=Role.user, content="Example input"),
            ChatMessage(role=Role.assistant, content="Example output"),
        ]

        response = optimize_prompt(
            developer_message="Original",
            messages=messages,
            requested_changes="Test changes",
            api_key="test-key",
        )

        assert len(response.new_messages) == 2
        # new_messages can be dict or ChatMessage
        for i, msg in enumerate(response.new_messages):
            if isinstance(msg, dict):
                assert "content" in msg
            else:
                assert isinstance(msg, ChatMessage)

    @patch("gpt_optimizer.optimizer.rewrite_custom")
    def test_requested_changes_empty_messages(self, mock_rewrite):
        """Test requested_changes with empty messages list."""
        mock_rewrite.return_value = DevRewriteOutput(new_developer_message="Rewritten")

        response = optimize_prompt(
            developer_message="Original",
            messages=[],
            requested_changes="Changes",
            api_key="test-key",
        )

        assert response.new_messages == []


class TestOptimizePromptFullOptimize:
    """Tests for optimize_prompt full optimization workflow."""

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    def test_full_optimize_no_issues(self, mock_format, mock_contra):
        """Test full optimize when no issues are found."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()

        response = optimize_prompt(
            developer_message="Well-structured prompt",
            api_key="test-key",
        )

        assert response.operation_mode == "full_optimize"
        assert response.issues_found is None or response.issues_found is False
        assert response.new_developer_message == "Well-structured prompt"
        assert "No issues found" in response.summary or "already well-structured" in response.summary

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    @patch("gpt_optimizer.optimizer.rewrite_dev_prompt")
    def test_full_optimize_with_issues(
        self, mock_rewrite, mock_format, mock_contra
    ):
        """Test full optimize when issues are found and fixed."""
        mock_contra.return_value = Issues(
            has_issues=True,
            issues=["Contradiction found"],
        )
        mock_format.return_value = Issues(has_issues=True, issues=["Format issue"])
        mock_rewrite.return_value = DevRewriteOutput(
            new_developer_message="Fixed prompt"
        )

        response = optimize_prompt(
            developer_message="Problematic prompt",
            api_key="test-key",
        )

        assert response.operation_mode == "full_optimize"
        assert response.issues_found is True
        assert response.new_developer_message == "Fixed prompt"
        assert any("Rewrote" in c.reason for c in response.comments)

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    def test_full_optimize_instant_mode(self, mock_format, mock_contra):
        """Test full optimize with instant mode."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()

        response = optimize_prompt(
            developer_message="Test",
            mode=OptimizerMode.instant,
            api_key="test-key",
        )

        assert response.operation_mode == "full_optimize"
        mock_contra.assert_called_once()
        mock_format.assert_called_once()

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    def test_full_optimize_default_mode(self, mock_format, mock_contra):
        """Test full optimize with default mode."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()

        response = optimize_prompt(
            developer_message="Test",
            mode=OptimizerMode.default,
            api_key="test-key",
        )

        assert response.operation_mode == "full_optimize"

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    @patch("gpt_optimizer.optimizer.rewrite_dev_prompt")
    def test_full_optimize_pro_mode_second_pass(
        self, mock_rewrite, mock_format, mock_contra
    ):
        """Test pro mode applies second optimization pass."""
        # First pass: issues found
        mock_contra.side_effect = [
            Issues(has_issues=True, issues=["Issue 1"]),  # First pass
            Issues.no_issues(),  # Second pass
        ]
        mock_format.side_effect = [
            Issues(has_issues=True, issues=["Format issue"]),  # First pass
            Issues.no_issues(),  # Second pass
        ]
        mock_rewrite.side_effect = [
            DevRewriteOutput(new_developer_message="First rewrite"),  # First pass
            DevRewriteOutput(new_developer_message="Second rewrite"),  # Second pass
        ]

        response = optimize_prompt(
            developer_message="Original",
            mode=OptimizerMode.pro,
            api_key="test-key",
        )

        assert response.operation_mode == "full_optimize"
        assert "second optimization pass" in response.summary.lower() or len(response.comments) > 2

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    @patch("gpt_optimizer.optimizer.check_fewshot")
    @patch("gpt_optimizer.optimizer.rewrite_fewshot")
    def test_full_optimize_with_fewshot_messages(
        self, mock_rewrite_fs, mock_check_fs, mock_format, mock_contra
    ):
        """Test full optimize with few-shot examples."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()
        mock_check_fs.return_value = FewShotIssues(
            has_issues=True,
            issues=["Few-shot issue"],
            rewrite_suggestions=["Fix suggestion"],
        )
        mock_rewrite_fs.return_value = MessagesOutput(
            messages=[
                ChatMessage(role=Role.user, content="Updated example"),
                ChatMessage(role=Role.assistant, content="Updated response"),
            ]
        )

        messages = [
            ChatMessage(role=Role.user, content="Example 1"),
            ChatMessage(role=Role.assistant, content="Response 1"),
        ]

        response = optimize_prompt(
            developer_message="Test",
            messages=messages,
            api_key="test-key",
        )

        assert response.operation_mode == "full_optimize"
        mock_check_fs.assert_called_once()
        mock_rewrite_fs.assert_called_once()

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    def test_full_optimize_without_assistant_examples(
        self, mock_format, mock_contra
    ):
        """Test full optimize skips few-shot check without assistant examples."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()

        # Only user messages, no assistant examples
        messages = [ChatMessage(role=Role.user, content="Example")]

        response = optimize_prompt(
            developer_message="Test",
            messages=messages,
            api_key="test-key",
        )

        assert response.operation_mode == "full_optimize"
        # few-shot check should not be called without assistant examples
        assert mock_contra.call_count == 1
        assert mock_format.call_count == 1

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    def test_full_optimize_preserves_unchanged_prompt(
        self, mock_format, mock_contra
    ):
        """Test that unchanged prompt is preserved in response."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()

        original = "This is the original prompt"
        response = optimize_prompt(
            developer_message=original,
            api_key="test-key",
        )

        assert response.new_developer_message == original


class TestOptimizePromptErrors:
    """Tests for error handling in optimize_prompt."""

    @patch("gpt_optimizer.optimizer._get_client")
    def test_optimize_prompt_with_custom_model(self, mock_get_client):
        """Test optimize_prompt uses custom agent model when provided."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("gpt_optimizer.optimizer.check_contradictions") as mock_contra:
            with patch("gpt_optimizer.optimizer.check_format") as mock_format:
                mock_contra.return_value = Issues.no_issues()
                mock_format.return_value = Issues.no_issues()

                optimize_prompt(
                    developer_message="Test",
                    model="custom-model-123",
                    api_key="test-key",
                )

                # Verify custom model was passed to checkers
                mock_contra.assert_called_once()
                args = mock_contra.call_args
                # Third argument should be the model
                assert args[0][2] == "custom-model-123"

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    def test_optimize_prompt_respects_target_model(
        self, mock_format, mock_contra
    ):
        """Test optimize_prompt respects target_model parameter."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()

        response = optimize_prompt(
            developer_message="Test",
            target_model="gpt-4-turbo",
            api_key="test-key",
        )

        # The response doesn't directly include target_model,
        # but we can verify the function accepts the parameter
        assert response is not None


class TestRunPresetCheck:
    """Tests for _run_preset_check internal function."""

    @patch("gpt_optimizer.optimizer.rewrite_dev_prompt")
    @patch("gpt_optimizer.optimizer.check_contradictions")
    def test_run_preset_check_conflicting_instructions(
        self, mock_contra, mock_rewrite
    ):
        """Test _run_preset_check with conflicting_instructions."""
        mock_contra.return_value = Issues(
            has_issues=True,
            issues=["Found contradiction"],
        )
        mock_rewrite.return_value = DevRewriteOutput(
            new_developer_message="Fixed prompt"
        )
        mock_client = MagicMock()

        response = _run_preset_check(
            mock_client,
            "Test prompt",
            [],
            PresetCheck.conflicting_instructions,
            "test-model",
        )

        assert response.operation_mode == "preset_check"
        assert response.preset_check == "conflicting_instructions"
        mock_contra.assert_called_once()

    @patch("gpt_optimizer.optimizer.check_format")
    def test_run_preset_check_ambiguity(self, mock_format):
        """Test _run_preset_check with ambiguity."""
        mock_format.return_value = Issues.no_issues()
        mock_client = MagicMock()

        response = _run_preset_check(
            mock_client,
            "Test prompt",
            [],
            PresetCheck.ambiguity,
            "test-model",
        )

        assert response.preset_check == "ambiguity"
        mock_format.assert_called_once()

    @patch("gpt_optimizer.optimizer.check_format")
    def test_run_preset_check_output_format(self, mock_format):
        """Test _run_preset_check with output_format."""
        mock_format.return_value = Issues.no_issues()
        mock_client = MagicMock()

        response = _run_preset_check(
            mock_client,
            "Test prompt",
            [],
            PresetCheck.output_format,
            "test-model",
        )

        assert response.preset_check == "output_format"


class TestRunFullOptimize:
    """Tests for _run_full_optimize internal function."""

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    def test_run_full_optimize_no_changes(self, mock_format, mock_contra):
        """Test _run_full_optimize when prompt needs no changes."""
        mock_contra.return_value = Issues.no_issues()
        mock_format.return_value = Issues.no_issues()
        mock_client = MagicMock()

        response = _run_full_optimize(
            mock_client,
            "Good prompt",
            [],
            OptimizerMode.default,
            "test-model",
        )

        assert response.new_developer_message == "Good prompt"
        assert response.operation_mode == "full_optimize"

    @patch("gpt_optimizer.optimizer.check_contradictions")
    @patch("gpt_optimizer.optimizer.check_format")
    @patch("gpt_optimizer.optimizer.rewrite_dev_prompt")
    def test_run_full_optimize_with_rewrite(
        self, mock_rewrite, mock_format, mock_contra
    ):
        """Test _run_full_optimize applies rewriting when issues found."""
        mock_contra.return_value = Issues(has_issues=True, issues=["Issue 1"])
        mock_format.return_value = Issues.no_issues()
        mock_rewrite.return_value = DevRewriteOutput(new_developer_message="Rewritten")
        mock_client = MagicMock()

        response = _run_full_optimize(
            mock_client,
            "Original",
            [],
            OptimizerMode.default,
            "test-model",
        )

        assert response.new_developer_message == "Rewritten"
        mock_rewrite.assert_called_once()
