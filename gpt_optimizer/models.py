"""Pydantic data models for the prompt optimization pipeline.

Mirrors the reverse-engineered schema from OpenAI's /v1/dashapi/optimize/promptv2.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class OptimizerMode(str, Enum):
    instant = "instant"
    default = "default"
    pro = "pro"


class PresetCheck(str, Enum):
    conflicting_instructions = "conflicting_instructions"
    ambiguity = "ambiguity"
    output_format = "output_format"


# ── Message Models ───────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Role
    content: str


# ── Agent Output Models ──────────────────────────────────────────────


class Issues(BaseModel):
    """Structured output returned by checker agents."""

    has_issues: bool
    issues: List[str]

    @classmethod
    def no_issues(cls) -> Issues:
        return cls(has_issues=False, issues=[])


class FewShotIssues(Issues):
    """Output for few-shot consistency checker, includes rewrite suggestions."""

    rewrite_suggestions: List[str] = Field(default_factory=list)

    @classmethod
    def no_issues(cls) -> FewShotIssues:
        return cls(has_issues=False, issues=[], rewrite_suggestions=[])


class DevRewriteOutput(BaseModel):
    """Rewriter returns the cleaned-up developer prompt."""

    new_developer_message: str


class MessagesOutput(BaseModel):
    """Structured output returned by the few-shot rewriter."""

    messages: list[ChatMessage]


# ── API Request / Response ───────────────────────────────────────────


class OptimizeRequest(BaseModel):
    """Request schema matching the reverse-engineered API."""

    developer_message: str
    messages: List[ChatMessage] = Field(default_factory=lambda: [ChatMessage(role=Role.user, content="")])
    model_name: str = "gpt-5.4"
    optimizer_mode: OptimizerMode = OptimizerMode.default
    tools: list = Field(default_factory=list)
    optimizing_for: str = "gpt-5.4"
    preset_check: Optional[PresetCheck] = None
    requested_changes: Optional[str] = None


class Comment(BaseModel):
    """A single comment/explanation about a change made."""

    kind: str  # "finding" or "explanation"
    reason: str
    location: Optional[dict] = None


class OptimizeResponse(BaseModel):
    """Response schema matching the reverse-engineered API."""

    comments: List[Comment] = Field(default_factory=list)
    issues_found: Optional[bool] = None
    new_developer_message: str
    new_messages: List[ChatMessage] = Field(default_factory=list)
    operation_mode: str = "full_optimize"
    preset_check: Optional[str] = None
    summary: str = ""
