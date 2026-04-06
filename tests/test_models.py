"""Tests for Pydantic models in gpt_optimizer.models."""

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


class TestRole:
    """Tests for Role enum."""

    def test_role_user(self):
        """Test Role.user value."""
        assert Role.user == "user"
        assert Role.user.value == "user"

    def test_role_assistant(self):
        """Test Role.assistant value."""
        assert Role.assistant == "assistant"
        assert Role.assistant.value == "assistant"


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_chat_message_user_role(self):
        """Test ChatMessage with user role."""
        msg = ChatMessage(role=Role.user, content="Hello, assistant!")
        assert msg.role == Role.user
        assert msg.content == "Hello, assistant!"

    def test_chat_message_assistant_role(self):
        """Test ChatMessage with assistant role."""
        msg = ChatMessage(role=Role.assistant, content="Hello, user!")
        assert msg.role == Role.assistant
        assert msg.content == "Hello, user!"

    def test_chat_message_with_string_role(self):
        """Test ChatMessage accepts string role and converts to enum."""
        msg = ChatMessage(role="user", content="Test message")
        assert msg.role == Role.user
        assert isinstance(msg.role, Role)

    def test_chat_message_empty_content(self):
        """Test ChatMessage with empty content."""
        msg = ChatMessage(role=Role.user, content="")
        assert msg.role == Role.user
        assert msg.content == ""

    def test_chat_message_long_content(self):
        """Test ChatMessage with long content."""
        long_content = "x" * 10000
        msg = ChatMessage(role=Role.user, content=long_content)
        assert len(msg.content) == 10000

    def test_chat_message_serialization(self):
        """Test ChatMessage can be serialized to dict."""
        msg = ChatMessage(role=Role.user, content="Test")
        data = msg.model_dump()
        assert data == {"role": "user", "content": "Test"}

    def test_chat_message_json_serialization(self):
        """Test ChatMessage can be serialized to JSON."""
        msg = ChatMessage(role=Role.user, content="Test")
        json_str = msg.model_dump_json()
        assert '"role":"user"' in json_str or '"role": "user"' in json_str
        assert '"content":"Test"' in json_str or '"content": "Test"' in json_str


class TestIssues:
    """Tests for Issues model."""

    def test_issues_no_issues_found(self):
        """Test Issues when no issues are found."""
        issues = Issues(has_issues=False, issues=[])
        assert issues.has_issues is False
        assert issues.issues == []

    def test_issues_with_issues_found(self):
        """Test Issues when issues are found."""
        issue_list = ["Issue 1", "Issue 2"]
        issues = Issues(has_issues=True, issues=issue_list)
        assert issues.has_issues is True
        assert issues.issues == issue_list

    def test_issues_no_issues_classmethod(self):
        """Test Issues.no_issues() classmethod."""
        issues = Issues.no_issues()
        assert issues.has_issues is False
        assert issues.issues == []
        assert isinstance(issues, Issues)

    def test_issues_single_issue(self):
        """Test Issues with a single issue."""
        issues = Issues(has_issues=True, issues=["Single issue"])
        assert issues.has_issues is True
        assert len(issues.issues) == 1

    def test_issues_multiple_issues(self):
        """Test Issues with multiple issues."""
        issue_list = ["Issue 1", "Issue 2", "Issue 3", "Issue 4"]
        issues = Issues(has_issues=True, issues=issue_list)
        assert len(issues.issues) == 4


class TestFewShotIssues:
    """Tests for FewShotIssues model."""

    def test_fewshot_issues_no_issues_found(self):
        """Test FewShotIssues when no issues are found."""
        issues = FewShotIssues(has_issues=False, issues=[], rewrite_suggestions=[])
        assert issues.has_issues is False
        assert issues.issues == []
        assert issues.rewrite_suggestions == []

    def test_fewshot_issues_with_issues(self):
        """Test FewShotIssues with issues and suggestions."""
        issues = FewShotIssues(
            has_issues=True,
            issues=["Issue 1"],
            rewrite_suggestions=["Suggestion 1", "Suggestion 2"],
        )
        assert issues.has_issues is True
        assert issues.issues == ["Issue 1"]
        assert issues.rewrite_suggestions == ["Suggestion 1", "Suggestion 2"]

    def test_fewshot_issues_no_issues_classmethod(self):
        """Test FewShotIssues.no_issues() classmethod."""
        issues = FewShotIssues.no_issues()
        assert issues.has_issues is False
        assert issues.issues == []
        assert issues.rewrite_suggestions == []
        assert isinstance(issues, FewShotIssues)

    def test_fewshot_issues_default_rewrite_suggestions(self):
        """Test FewShotIssues with default empty rewrite_suggestions."""
        issues = FewShotIssues(has_issues=False, issues=[])
        assert issues.rewrite_suggestions == []

    def test_fewshot_issues_inherits_from_issues(self):
        """Test FewShotIssues is subclass of Issues."""
        issues = FewShotIssues.no_issues()
        assert isinstance(issues, Issues)

    def test_fewshot_issues_serialization(self):
        """Test FewShotIssues serialization."""
        issues = FewShotIssues(
            has_issues=True,
            issues=["Issue 1"],
            rewrite_suggestions=["Suggestion 1"],
        )
        data = issues.model_dump()
        assert data["has_issues"] is True
        assert data["issues"] == ["Issue 1"]
        assert data["rewrite_suggestions"] == ["Suggestion 1"]


class TestDevRewriteOutput:
    """Tests for DevRewriteOutput model."""

    def test_dev_rewrite_output_with_message(self):
        """Test DevRewriteOutput with a rewritten message."""
        output = DevRewriteOutput(new_developer_message="Rewritten prompt")
        assert output.new_developer_message == "Rewritten prompt"

    def test_dev_rewrite_output_empty_message(self):
        """Test DevRewriteOutput with empty message."""
        output = DevRewriteOutput(new_developer_message="")
        assert output.new_developer_message == ""

    def test_dev_rewrite_output_long_message(self):
        """Test DevRewriteOutput with long message."""
        long_message = "x" * 50000
        output = DevRewriteOutput(new_developer_message=long_message)
        assert len(output.new_developer_message) == 50000

    def test_dev_rewrite_output_serialization(self):
        """Test DevRewriteOutput serialization."""
        output = DevRewriteOutput(new_developer_message="Test rewrite")
        data = output.model_dump()
        assert data == {"new_developer_message": "Test rewrite"}


class TestMessagesOutput:
    """Tests for MessagesOutput model."""

    def test_messages_output_empty(self):
        """Test MessagesOutput with empty messages."""
        output = MessagesOutput(messages=[])
        assert output.messages == []

    def test_messages_output_single_message(self):
        """Test MessagesOutput with single message."""
        msg = ChatMessage(role=Role.user, content="Test")
        output = MessagesOutput(messages=[msg])
        assert len(output.messages) == 1
        assert output.messages[0].role == Role.user

    def test_messages_output_multiple_messages(self):
        """Test MessagesOutput with multiple messages."""
        messages = [
            ChatMessage(role=Role.user, content="User message"),
            ChatMessage(role=Role.assistant, content="Assistant message"),
            ChatMessage(role=Role.user, content="Another user message"),
        ]
        output = MessagesOutput(messages=messages)
        assert len(output.messages) == 3
        assert output.messages[0].role == Role.user
        assert output.messages[1].role == Role.assistant

    def test_messages_output_serialization(self):
        """Test MessagesOutput serialization."""
        messages = [ChatMessage(role=Role.user, content="Test")]
        output = MessagesOutput(messages=messages)
        data = output.model_dump()
        assert "messages" in data
        assert len(data["messages"]) == 1


class TestComment:
    """Tests for Comment model."""

    def test_comment_finding(self):
        """Test Comment with kind='finding'."""
        comment = Comment(kind="finding", reason="This is an issue")
        assert comment.kind == "finding"
        assert comment.reason == "This is an issue"
        assert comment.location is None

    def test_comment_explanation(self):
        """Test Comment with kind='explanation'."""
        comment = Comment(kind="explanation", reason="This was changed because...")
        assert comment.kind == "explanation"
        assert comment.reason == "This was changed because..."

    def test_comment_with_location(self):
        """Test Comment with location metadata."""
        location = {"line": 1, "column": 5}
        comment = Comment(kind="finding", reason="Issue", location=location)
        assert comment.location == location

    def test_comment_serialization(self):
        """Test Comment serialization."""
        comment = Comment(kind="finding", reason="Test issue")
        data = comment.model_dump()
        assert data["kind"] == "finding"
        assert data["reason"] == "Test issue"


class TestOptimizerMode:
    """Tests for OptimizerMode enum."""

    def test_optimizer_mode_instant(self):
        """Test OptimizerMode.instant."""
        assert OptimizerMode.instant == "instant"

    def test_optimizer_mode_default(self):
        """Test OptimizerMode.default."""
        assert OptimizerMode.default == "default"

    def test_optimizer_mode_pro(self):
        """Test OptimizerMode.pro."""
        assert OptimizerMode.pro == "pro"


class TestPresetCheck:
    """Tests for PresetCheck enum."""

    def test_preset_check_conflicting_instructions(self):
        """Test PresetCheck.conflicting_instructions."""
        assert PresetCheck.conflicting_instructions == "conflicting_instructions"

    def test_preset_check_ambiguity(self):
        """Test PresetCheck.ambiguity."""
        assert PresetCheck.ambiguity == "ambiguity"

    def test_preset_check_output_format(self):
        """Test PresetCheck.output_format."""
        assert PresetCheck.output_format == "output_format"


class TestOptimizeRequest:
    """Tests for OptimizeRequest model."""

    def test_optimize_request_minimal(self):
        """Test OptimizeRequest with only required developer_message."""
        request = OptimizeRequest(developer_message="Optimize this prompt")
        assert request.developer_message == "Optimize this prompt"
        assert len(request.messages) == 1
        assert request.messages[0].role == Role.user
        assert request.messages[0].content == ""
        assert request.model_name == "gpt-5.4"
        assert request.optimizer_mode == OptimizerMode.default
        assert request.tools == []
        assert request.optimizing_for == "gpt-5.4"
        assert request.preset_check is None
        assert request.requested_changes is None

    def test_optimize_request_with_messages(self):
        """Test OptimizeRequest with custom messages."""
        messages = [
            ChatMessage(role=Role.user, content="Example input"),
            ChatMessage(role=Role.assistant, content="Example output"),
        ]
        request = OptimizeRequest(developer_message="Test", messages=messages)
        assert request.messages == messages

    def test_optimize_request_with_optimizer_mode(self):
        """Test OptimizeRequest with custom optimizer_mode."""
        request = OptimizeRequest(
            developer_message="Test",
            optimizer_mode=OptimizerMode.pro,
        )
        assert request.optimizer_mode == OptimizerMode.pro

    def test_optimize_request_with_preset_check(self):
        """Test OptimizeRequest with preset_check."""
        request = OptimizeRequest(
            developer_message="Test",
            preset_check=PresetCheck.ambiguity,
        )
        assert request.preset_check == PresetCheck.ambiguity

    def test_optimize_request_with_requested_changes(self):
        """Test OptimizeRequest with requested_changes."""
        request = OptimizeRequest(
            developer_message="Test",
            requested_changes="Make it more concise",
        )
        assert request.requested_changes == "Make it more concise"

    def test_optimize_request_with_tools(self):
        """Test OptimizeRequest with tools."""
        tools = [{"type": "function", "function": {"name": "test"}}]
        request = OptimizeRequest(developer_message="Test", tools=tools)
        assert request.tools == tools

    def test_optimize_request_with_custom_model(self):
        """Test OptimizeRequest with custom model_name."""
        request = OptimizeRequest(
            developer_message="Test",
            model_name="gpt-4",
            optimizing_for="gpt-4",
        )
        assert request.model_name == "gpt-4"
        assert request.optimizing_for == "gpt-4"

    def test_optimize_request_full_params(self):
        """Test OptimizeRequest with all parameters."""
        messages = [ChatMessage(role=Role.user, content="Example")]
        request = OptimizeRequest(
            developer_message="Optimize",
            messages=messages,
            model_name="gpt-4",
            optimizer_mode=OptimizerMode.instant,
            tools=[],
            optimizing_for="gpt-4",
            preset_check=PresetCheck.output_format,
            requested_changes="Be more specific",
        )
        assert request.developer_message == "Optimize"
        assert request.messages == messages
        assert request.model_name == "gpt-4"
        assert request.optimizer_mode == OptimizerMode.instant
        assert request.optimizing_for == "gpt-4"
        assert request.preset_check == PresetCheck.output_format
        assert request.requested_changes == "Be more specific"

    def test_optimize_request_default_empty_messages(self):
        """Test OptimizeRequest default empty messages list."""
        request = OptimizeRequest(developer_message="Test", messages=[])
        assert request.messages == []


class TestOptimizeResponse:
    """Tests for OptimizeResponse model."""

    def test_optimize_response_minimal(self):
        """Test OptimizeResponse with minimal required fields."""
        response = OptimizeResponse(new_developer_message="Optimized prompt")
        assert response.new_developer_message == "Optimized prompt"
        assert response.comments == []
        assert response.issues_found is None
        assert response.new_messages == []
        assert response.operation_mode == "full_optimize"
        assert response.preset_check is None
        assert response.summary == ""

    def test_optimize_response_with_comments(self):
        """Test OptimizeResponse with comments."""
        comments = [
            Comment(kind="finding", reason="Found issue"),
            Comment(kind="explanation", reason="Fixed it"),
        ]
        response = OptimizeResponse(
            new_developer_message="Test",
            comments=comments,
        )
        assert len(response.comments) == 2
        assert response.comments[0].kind == "finding"

    def test_optimize_response_with_messages(self):
        """Test OptimizeResponse with new_messages."""
        messages = [ChatMessage(role=Role.user, content="Example")]
        response = OptimizeResponse(
            new_developer_message="Test",
            new_messages=messages,
        )
        assert response.new_messages == messages

    def test_optimize_response_with_issues_found_true(self):
        """Test OptimizeResponse with issues_found=True."""
        response = OptimizeResponse(
            new_developer_message="Test",
            issues_found=True,
        )
        assert response.issues_found is True

    def test_optimize_response_with_issues_found_false(self):
        """Test OptimizeResponse with issues_found=False."""
        response = OptimizeResponse(
            new_developer_message="Test",
            issues_found=False,
        )
        assert response.issues_found is False

    def test_optimize_response_full_params(self):
        """Test OptimizeResponse with all parameters."""
        comments = [Comment(kind="finding", reason="Test")]
        messages = [ChatMessage(role=Role.user, content="Example")]
        response = OptimizeResponse(
            comments=comments,
            issues_found=True,
            new_developer_message="Optimized",
            new_messages=messages,
            operation_mode="preset_check",
            preset_check="ambiguity",
            summary="Found and fixed ambiguity issues",
        )
        assert response.comments == comments
        assert response.issues_found is True
        assert response.new_developer_message == "Optimized"
        assert response.new_messages == messages
        assert response.operation_mode == "preset_check"
        assert response.preset_check == "ambiguity"
        assert response.summary == "Found and fixed ambiguity issues"

    def test_optimize_response_serialization(self):
        """Test OptimizeResponse serialization."""
        response = OptimizeResponse(
            new_developer_message="Test",
            summary="Test summary",
        )
        data = response.model_dump()
        assert data["new_developer_message"] == "Test"
        assert data["summary"] == "Test summary"
        assert data["operation_mode"] == "full_optimize"

    def test_optimize_response_json_serialization(self):
        """Test OptimizeResponse JSON serialization."""
        response = OptimizeResponse(
            new_developer_message="Test",
            issues_found=True,
        )
        json_str = response.model_dump_json()
        assert '"new_developer_message":"Test"' in json_str or '"new_developer_message": "Test"' in json_str
        assert '"issues_found":true' in json_str or '"issues_found": true' in json_str
