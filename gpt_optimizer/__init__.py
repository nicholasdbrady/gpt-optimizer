"""GPT Optimizer — Programmatic prompt optimization.

Usage:
    from gpt_optimizer import optimize_prompt
    result = optimize_prompt("You are a helpful assistant.")
"""

from .optimizer import optimize_prompt, optimize_from_request
from .models import (
    ChatMessage,
    OptimizeRequest,
    OptimizeResponse,
    OptimizerMode,
    PresetCheck,
    Role,
)

__all__ = [
    "optimize_prompt",
    "optimize_from_request",
    "ChatMessage",
    "OptimizeRequest",
    "OptimizeResponse",
    "OptimizerMode",
    "PresetCheck",
    "Role",
]