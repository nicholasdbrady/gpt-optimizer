"""Golden-example evaluation tests for the prompt optimizer.

Loads test cases from golden_examples.json and verifies the optimizer
detects the expected issue types using mocked agent responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_optimizer.models import (
    ChatMessage,
    DevRewriteOutput,
    FewShotIssues,
    Issues,
    MessagesOutput,
    Role,
)
from gpt_optimizer.optimizer import optimize_prompt

GOLDEN_PATH = Path(__file__).parent / "golden_examples.json"


def load_golden_examples() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


GOLDEN_EXAMPLES = load_golden_examples()


def _build_messages(raw: list[dict]) -> list[ChatMessage]:
    return [ChatMessage(role=Role(m["role"]), content=m["content"]) for m in raw]


def _mock_run_agent(example: dict):
    """Build a side_effect for run_agent that returns the right model per call.

    The optimizer calls run_agent through the checker/rewriter convenience
    functions. We inspect the *output_type* argument (positional arg 3) to
    decide which mock response to return.
    """
    expected = example["expected"]

    cd_has = bool(expected["contradiction_issues"])
    cd_issues = Issues(
        has_issues=cd_has,
        issues=[expected["contradiction_issues"]] if cd_has else [],
    )

    fi_has = bool(expected["format_issues"])
    fi_issues = Issues(
        has_issues=fi_has,
        issues=[expected["format_issues"]] if fi_has else [],
    )

    fs_has = bool(expected["few_shot_issues"])
    fs_issues = FewShotIssues(
        has_issues=fs_has,
        issues=[expected["few_shot_issues"]] if fs_has else [],
        rewrite_suggestions=["Rewrite to fix issue."] if fs_has else [],
    )

    rewrite_dev = DevRewriteOutput(
        new_developer_message=example["input"]["developer_message"] + "\n[OPTIMIZED]"
        if expected["has_changes"]
        else example["input"]["developer_message"],
    )

    rewrite_msgs = MessagesOutput(
        messages=_build_messages(example["input"]["messages"])
    )

    # Track which checker has been called to return in order
    call_counter = {"issues": 0}

    def _side_effect(_client, _instructions, _user_input, output_type, _model=None):
        if output_type is Issues:
            idx = call_counter["issues"]
            call_counter["issues"] += 1
            # First Issues call = contradiction checker, second = format checker
            return cd_issues if idx == 0 else fi_issues
        if output_type is FewShotIssues:
            return fs_issues
        if output_type is DevRewriteOutput:
            return rewrite_dev
        if output_type is MessagesOutput:
            return rewrite_msgs
        raise ValueError(f"Unexpected output_type: {output_type}")

    return _side_effect



@pytest.mark.parametrize(
    "example",
    GOLDEN_EXAMPLES,
    ids=[e["id"] for e in GOLDEN_EXAMPLES],
)
def test_golden_example(example: dict):
    """Verify the optimizer detects expected issue types for each golden case."""
    messages = _build_messages(example["input"]["messages"])
    expected = example["expected"]

    with patch("gpt_optimizer.agents.run_agent", side_effect=_mock_run_agent(example)):
        response = optimize_prompt(
            developer_message=example["input"]["developer_message"],
            messages=messages,
            api_key="test-key",
        )

    comment_text = " ".join(c.reason for c in response.comments)

    # Verify changes were (or weren't) made
    has_findings = any(c.kind == "finding" for c in response.comments)
    if expected["has_changes"]:
        assert has_findings, (
            f"[{example['id']}] Expected issues to be found but no findings reported"
        )
    else:
        assert not has_findings, (
            f"[{example['id']}] Expected no issues but findings were reported"
        )

    # Verify contradiction detection
    if expected["contradiction_issues"]:
        assert any("Contradiction" in c.reason for c in response.comments), (
            f"[{example['id']}] Expected contradiction finding in comments"
        )
    else:
        assert not any("Contradiction" in c.reason for c in response.comments), (
            f"[{example['id']}] Unexpected contradiction finding in comments"
        )

    # Verify format detection
    if expected["format_issues"]:
        assert any("Format" in c.reason for c in response.comments), (
            f"[{example['id']}] Expected format finding in comments"
        )
    else:
        assert not any("Format" in c.reason for c in response.comments), (
            f"[{example['id']}] Unexpected format finding in comments"
        )

    # Verify few-shot detection
    if expected["few_shot_issues"]:
        assert any("Few-shot" in c.reason for c in response.comments), (
            f"[{example['id']}] Expected few-shot finding in comments"
        )
    else:
        assert not any("Few-shot" in c.reason for c in response.comments), (
            f"[{example['id']}] Unexpected few-shot finding in comments"
        )



def test_golden_dataset_integrity():
    """Verify the golden dataset is well-formed and covers required categories."""
    examples = load_golden_examples()

    assert len(examples) >= 15, f"Expected >= 15 examples, got {len(examples)}"

    focus_counts: dict[str, int] = {}
    ids_seen: set[str] = set()

    for ex in examples:
        assert "id" in ex and "focus" in ex and "input" in ex and "expected" in ex
        assert ex["id"] not in ids_seen, f"Duplicate id: {ex['id']}"
        ids_seen.add(ex["id"])

        assert "developer_message" in ex["input"]
        assert "messages" in ex["input"]

        exp = ex["expected"]
        assert "has_changes" in exp
        assert "contradiction_issues" in exp
        assert "format_issues" in exp
        assert "few_shot_issues" in exp

        focus_counts[ex["focus"]] = focus_counts.get(ex["focus"], 0) + 1

    assert focus_counts.get("contradiction_issues", 0) >= 4
    assert focus_counts.get("format_issues", 0) >= 3
    assert focus_counts.get("few_shot_issues", 0) >= 3
    assert focus_counts.get("no_issues", 0) >= 3
    assert focus_counts.get("combination", 0) >= 2
