"""Core optimization workflow.

Orchestrates the multi-agent pipeline:
  1. Parallel issue detection (contradiction, format, few-shot)
  2. Conditional rewriting (dev prompt, few-shot examples)

Supports three modes: instant (fast), default (balanced), pro (thorough).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from .config import DEFAULT_MODEL, get_openai_client
from .models import (
    ChatMessage,
    Comment,
    FewShotIssues,
    Issues,
    OptimizeRequest,
    OptimizeResponse,
    OptimizerMode,
    PresetCheck,
    Role,
)
from .agents import (
    check_contradictions,
    check_format,
    check_fewshot,
    rewrite_dev_prompt,
    rewrite_fewshot,
    rewrite_custom,
)


def _normalize_messages(messages: list[Any]) -> list[dict[str, str]]:
    """Convert message models to plain dicts."""
    result = []
    for m in messages:
        if hasattr(m, "model_dump"):
            result.append(m.model_dump())
        elif isinstance(m, dict) and "role" in m and "content" in m:
            result.append({"role": str(m["role"]), "content": str(m["content"])})
    return result


def _get_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client using the best available auth method."""
    return get_openai_client(api_key=api_key)


def optimize_prompt(
    developer_message: str,
    messages: list[ChatMessage] | None = None,
    mode: OptimizerMode = OptimizerMode.default,
    preset_check: PresetCheck | None = None,
    requested_changes: str | None = None,
    model: str | None = None,
    target_model: str = "gpt-5.4",
    api_key: str | None = None,
) -> OptimizeResponse:
    """
    Optimize a prompt using the multi-agent workflow.

    Args:
        developer_message: The system/developer prompt to optimize.
        messages: Optional few-shot example messages.
        mode: Speed/depth: instant, default, or pro.
        preset_check: Run a single targeted check only.
        requested_changes: Free-text custom optimization instruction.
        model: Override the agent model (default: gpt-5.4).
        target_model: Target model to optimize for.
        api_key: Override the OpenAI API key.

    Returns:
        OptimizeResponse with the optimized prompt and change details.
    """
    client = _get_client(api_key)
    agent_model = model or DEFAULT_MODEL
    messages = messages or []

    # Handle custom requested_changes
    if requested_changes:
        result = rewrite_custom(client, developer_message, requested_changes, agent_model)
        return OptimizeResponse(
            comments=[Comment(kind="explanation", reason=f"Applied requested changes: {requested_changes}")],
            new_developer_message=result.new_developer_message,
            new_messages=_normalize_messages(messages) if messages else [],
            operation_mode="custom",
            summary=f"Applied custom optimization: {requested_changes}",
        )

    # Handle targeted preset checks
    if preset_check:
        return _run_preset_check(client, developer_message, messages, preset_check, agent_model)

    # Full optimization workflow
    return _run_full_optimize(client, developer_message, messages, mode, agent_model)


def _run_preset_check(
    client: OpenAI,
    developer_message: str,
    messages: list[ChatMessage],
    preset_check: PresetCheck,
    model: str,
) -> OptimizeResponse:
    """Run a single targeted check and fix."""
    comments: list[Comment] = []

    if preset_check == PresetCheck.conflicting_instructions:
        issues = check_contradictions(client, developer_message, model)
    elif preset_check == PresetCheck.ambiguity:
        # Ambiguity uses the format checker with a broader lens
        issues = check_format(client, developer_message, model)
    elif preset_check == PresetCheck.output_format:
        issues = check_format(client, developer_message, model)
    else:
        issues = Issues.no_issues()

    for issue in issues.issues:
        comments.append(Comment(kind="finding", reason=issue))

    new_dev_message = developer_message
    if issues.has_issues:
        rewrite = rewrite_dev_prompt(
            client, developer_message, issues, Issues.no_issues(), model
        )
        new_dev_message = rewrite.new_developer_message
        comments.append(Comment(
            kind="explanation",
            reason=f"Rewrote prompt to resolve {len(issues.issues)} issue(s).",
        ))

    issue_count = len(issues.issues)
    return OptimizeResponse(
        comments=comments,
        issues_found=issues.has_issues,
        new_developer_message=new_dev_message,
        new_messages=_normalize_messages(messages),
        operation_mode="preset_check",
        preset_check=preset_check.value,
        summary=f"{preset_check.value} check found {issue_count} issue(s)."
        if issues.has_issues
        else f"{preset_check.value} check found no issues.",
    )


def _run_full_optimize(
    client: OpenAI,
    developer_message: str,
    messages: list[ChatMessage],
    mode: OptimizerMode,
    model: str,
) -> OptimizeResponse:
    """Run the full parallel check → rewrite pipeline."""
    comments: list[Comment] = []

    # Phase 1: parallel issue detection using threads
    has_examples = any(m.role == Role.assistant for m in messages)

    with ThreadPoolExecutor(max_workers=3) as executor:
        cd_future = executor.submit(check_contradictions, client, developer_message, model)
        fi_future = executor.submit(check_format, client, developer_message, model)
        fs_future = None
        if has_examples:
            user_examples = [m.content for m in messages if m.role == Role.user]
            assistant_examples = [m.content for m in messages if m.role == Role.assistant]
            fs_future = executor.submit(
                check_fewshot, client, developer_message, user_examples, assistant_examples, model
            )

    cd_issues: Issues = cd_future.result()
    fi_issues: Issues = fi_future.result()
    fs_issues: FewShotIssues = fs_future.result() if fs_future else FewShotIssues.no_issues()

    for issue in cd_issues.issues:
        comments.append(Comment(kind="finding", reason=f"Contradiction: {issue}"))
    for issue in fi_issues.issues:
        comments.append(Comment(kind="finding", reason=f"Format: {issue}"))
    for issue in fs_issues.issues:
        comments.append(Comment(kind="finding", reason=f"Few-shot: {issue}"))

    # Phase 2: conditional rewriting
    final_prompt = developer_message
    if cd_issues.has_issues or fi_issues.has_issues:
        rewrite = rewrite_dev_prompt(client, developer_message, cd_issues, fi_issues, model)
        final_prompt = rewrite.new_developer_message
        comments.append(Comment(
            kind="explanation",
            reason="Rewrote developer prompt to resolve detected issues.",
        ))

    final_messages = messages
    if fs_issues.has_issues:
        mr_result = rewrite_fewshot(
            client, final_prompt, _normalize_messages(messages), fs_issues, model
        )
        final_messages = mr_result.messages
        comments.append(Comment(
            kind="explanation",
            reason="Rewrote few-shot examples to comply with updated prompt.",
        ))

    # Pro mode: run a second optimization pass
    if mode == OptimizerMode.pro and final_prompt != developer_message:
        cd2 = check_contradictions(client, final_prompt, model)
        fi2 = check_format(client, final_prompt, model)
        if cd2.has_issues or fi2.has_issues:
            rewrite2 = rewrite_dev_prompt(client, final_prompt, cd2, fi2, model)
            final_prompt = rewrite2.new_developer_message
            comments.append(Comment(
                kind="explanation",
                reason="Pro mode: applied second optimization pass.",
            ))

    total_issues = len(cd_issues.issues) + len(fi_issues.issues) + len(fs_issues.issues)
    has_changes = final_prompt != developer_message or final_messages is not messages

    return OptimizeResponse(
        comments=comments,
        issues_found=total_issues > 0 if total_issues else None,
        new_developer_message=final_prompt,
        new_messages=_normalize_messages(final_messages),
        operation_mode="full_optimize",
        summary=f"Applied a full optimize pass with {len(comments)} change(s)."
        if has_changes
        else "No issues found; prompt is already well-structured.",
    )


def optimize_from_request(request: OptimizeRequest, api_key: str | None = None) -> OptimizeResponse:
    """Convenience wrapper that takes an OptimizeRequest object."""
    return optimize_prompt(
        developer_message=request.developer_message,
        messages=request.messages,
        mode=request.optimizer_mode,
        preset_check=request.preset_check,
        requested_changes=request.requested_changes,
        target_model=request.optimizing_for,
        api_key=api_key,
    )
