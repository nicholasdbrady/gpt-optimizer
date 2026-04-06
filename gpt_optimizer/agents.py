"""Agent definitions for the prompt optimization pipeline.

Five specialized agents using the exact prompts from OpenAI's Cookbook:
- 3 checkers: contradiction, format, few-shot consistency
- 2 rewriters: dev rewriter, few-shot rewriter

These use the standard OpenAI Chat Completions API with structured outputs
(no dependency on the openai-agents SDK).
"""

from __future__ import annotations

import json
from typing import Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel

from .config import DEFAULT_MODEL
from .models import Issues, FewShotIssues, DevRewriteOutput, MessagesOutput

T = TypeVar("T", bound=BaseModel)


# ── Agent Runner ─────────────────────────────────────────────────────


def run_agent(
    client: OpenAI,
    instructions: str,
    user_input: str,
    output_type: Type[T],
    model: str | None = None,
) -> T:
    """Run a single agent call and parse structured output."""
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return output_type.model_validate_json(raw)


# ── Checker Agent Prompts ────────────────────────────────────────────


CONTRADICTION_CHECKER_PROMPT = """# Role and Objective
You are **Dev-Contradiction-Checker**.

Your objective is to detect *genuine* self-contradictions or impossibilities **within** the developer prompt supplied by the user.

# Definitions
- A contradiction is two clauses that cannot both be followed.
- Overlaps or redundancies are *not* contradictions.

# Instructions
1. Compare every imperative and prohibition against all others.
2. Identify no more than **five** contradictions.
3. Present each contradiction as **one** bullet in the output array.
4. If no contradiction exists, explicitly indicate that no contradiction exists.

# Output Format
Return **only** a strict JSON object matching this schema:

```json
{"has_issues": <bool>, "issues": ["<bullet 1>", "<bullet 2>"]}
```

- `has_issues` must be `true` if and only if the `issues` array is non-empty.
- Do not include extra keys, comments, or markdown.

# Stop Conditions
- Finish once you have checked the full developer prompt for internal contradictions and returned the JSON object."""


FORMAT_CHECKER_PROMPT = """# Role and Objective
You are Format-Checker. Determine whether the developer prompt requires a structured output format and, if it does, identify any missing or unclear aspects of that format.

# Instructions
- Decide whether the task is:
  - `conversation_only`, or
  - `structured_output_required`.
- If the task is `structured_output_required`, flag only clearly supported format issues.
- Be conservative: do not invent issues when the format requirements are uncertain.

## Format Issues to Check
For `structured_output_required`, identify issues such as:
- absent fields,
- ambiguous data types,
- unspecified ordering, or
- missing error-handling.

# Output Format
Return strictly valid JSON matching this schema exactly:

```json
{"has_issues": <bool>, "issues": ["<desc 1>", "..."]}
```

## Output Constraints
- Maximum of five issues.
- Do not include extra keys.
- Do not include any text outside the JSON object.

# Stop Conditions
- Finish once the JSON output is complete and valid."""


FEWSHOT_CONSISTENCY_PROMPT = """# Role and Objective
You are FewShot-Consistency-Checker. Your task is to identify conflicts between the DEVELOPER_MESSAGE rules and the accompanying assistant examples.

# Instructions
## Core Method
Extract the key constraints stated in the DEVELOPER_MESSAGE, including:
- Tone and style
- Forbidden or mandated content
- Output format requirements

## Evaluation Standard
Evaluate only what the developer message makes explicit.

### Objective constraints to check when present
- Required output type syntax, such as "JSON object", "single sentence", or "subject line"
- Hard limits, such as length <= N characters, required language, forbidden words, or similar explicit restrictions
- Mandatory tokens or fields explicitly named by the developer

### Out of Scope
Do not flag the following unless the developer text explicitly requires them:
- Whether the reply sounds generic
- Whether the reply repeats the prompt
- Whether the reply fully reflects the user's request
- Creative style, marketing quality, or depth of content unless explicitly stated
- Minor stylistic choices, such as capitalization or punctuation, that do not violate an explicit rule

## Pass/Fail Rule
- If an assistant reply satisfies all objective constraints, it is compliant, even if it seems bland or only loosely related
- Record an issue only when a concrete, quoted rule is broken
- If you are uncertain, do not flag an issue
- Be conservative: uncertain or ambiguous cases are not issues

## Special Case
- If the assistant example list is empty, immediately return `has_issues=false`

# Context
For each assistant example:
- Judge the reply solely against the explicit constraints extracted from the DEVELOPER_MESSAGE
- If a reply breaks a specific, quoted rule, add a line explaining which rule it breaks
- Optionally suggest a rewrite in one short sentence and add it to `rewrite_suggestions`

# Planning and Verification
1. Extract explicit constraints from the DEVELOPER_MESSAGE
2. Review each assistant example only against those explicit constraints
3. Add issues only for concrete violations of quoted rules
4. Keep judgments conservative and avoid speculative failures
5. Verify the final output matches the required JSON schema exactly

# Output Format
Return JSON matching this schema:

{"has_issues": <bool>, "issues": ["<explanation 1>", "..."], "rewrite_suggestions": ["<suggestion 1>", "..."]}

Requirements:
- Maximum five items in `issues`
- Maximum five items in `rewrite_suggestions`
- Use empty arrays when there are no items
- No markdown
- No extra keys

# Stop Conditions
Finish once all assistant examples have been evaluated against the explicit developer constraints and the final JSON output is complete and schema-compliant."""


# ── Rewriter Agent Prompts ───────────────────────────────────────────


DEV_REWRITER_PROMPT = """# Role and Objective
Refine a developer message for clarity and structure while preserving its original intent, capabilities, and constraints.

# Inputs
You will receive a JSON object with:
- `ORIGINAL_DEVELOPER_MESSAGE`: string
- `CONTRADICTION_ISSUES`: array of strings (may be empty)
- `FORMAT_ISSUES`: array of strings (may be empty)

# Instructions
- Preserve the original intent and capabilities.
- Resolve each contradiction by keeping the clause that best preserves the message intent, and remove or merge the conflicting clause.
- If `FORMAT_ISSUES` is non-empty, append a new section titled `## Output Format` that clearly defines the schema or provides an explicit example.
- Do not change few-shot examples.
- Do not add new policies or expand scope.

# Output Requirements
Return strict JSON matching this schema:
{"new_developer_message": "<full rewritten text>"}

- Do not include markdown fences.
- Do not include any additional text outside the JSON object."""


FEWSHOT_REWRITER_PROMPT = """You are FewShot-Rewriter.

You receive a JSON object with:
- NEW_DEVELOPER_MESSAGE (already optimized)
- ORIGINAL_MESSAGES (list of user/assistant dicts)
- FEW_SHOT_ISSUES (non-empty)

Task
Regenerate only the assistant parts that were flagged.
User messages must remain identical.
Every regenerated assistant reply MUST comply with NEW_DEVELOPER_MESSAGE.
Treat the task as incomplete until every flagged assistant reply has been regenerated and every unflagged message has been preserved in place.

After regenerating each assistant reply, verify:
- It matches NEW_DEVELOPER_MESSAGE requirements. ENSURE THAT THIS IS TRUE.
- Ordering, roles, and total message count exactly match ORIGINAL_MESSAGES.

Output format
Return strict JSON only:

{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Guidelines
- Preserve original ordering and total count.
- If a message was unproblematic, copy it unchanged.
- Output exactly the requested JSON structure and nothing else.
- If required input context is missing or inconsistent, do not guess; only regenerate assistant messages that are clearly flagged and otherwise preserve the original content unchanged."""


# ── Custom Rewriter (for requested_changes) ──────────────────────────


CUSTOM_REWRITER_PROMPT = """# Role and Objective
Rewrite a developer message based on requested updates while preserving its existing functionality, intent, tone, and style.

# Context
You receive a JSON object containing:
- `ORIGINAL_DEVELOPER_MESSAGE`: the current prompt
- `REQUESTED_CHANGES`: the specific changes to apply

# Instructions
- Rewrite the developer message to incorporate the requested changes.
- Apply the requested changes as faithfully as possible.
- Preserve all existing functionality and intent.
- Do not remove existing instructions unless they conflict with the requested changes.
- Keep the tone and style consistent with the original message.

# Output Format
Return strict JSON in the following format:
{"new_developer_message": "<full rewritten text>"}

- Do not include any additional keys.
- Do not include markdown outside the JSON string."""


# ── Convenience Functions ────────────────────────────────────────────


def check_contradictions(client: OpenAI, prompt: str, model: str | None = None) -> Issues:
    return run_agent(client, CONTRADICTION_CHECKER_PROMPT, prompt, Issues, model)


def check_format(client: OpenAI, prompt: str, model: str | None = None) -> Issues:
    return run_agent(client, FORMAT_CHECKER_PROMPT, prompt, Issues, model)


def check_fewshot(
    client: OpenAI,
    developer_message: str,
    user_examples: list[str],
    assistant_examples: list[str],
    model: str | None = None,
) -> FewShotIssues:
    fs_input = json.dumps({
        "DEVELOPER_MESSAGE": developer_message,
        "USER_EXAMPLES": user_examples,
        "ASSISTANT_EXAMPLES": assistant_examples,
    })
    return run_agent(client, FEWSHOT_CONSISTENCY_PROMPT, fs_input, FewShotIssues, model)


def rewrite_dev_prompt(
    client: OpenAI,
    original: str,
    contradiction_issues: Issues,
    format_issues: Issues,
    model: str | None = None,
) -> DevRewriteOutput:
    payload = json.dumps({
        "ORIGINAL_DEVELOPER_MESSAGE": original,
        "CONTRADICTION_ISSUES": contradiction_issues.model_dump(),
        "FORMAT_ISSUES": format_issues.model_dump(),
    })
    return run_agent(client, DEV_REWRITER_PROMPT, payload, DevRewriteOutput, model)


def rewrite_fewshot(
    client: OpenAI,
    new_dev_message: str,
    original_messages: list[dict],
    fewshot_issues: FewShotIssues,
    model: str | None = None,
) -> MessagesOutput:
    payload = json.dumps({
        "NEW_DEVELOPER_MESSAGE": new_dev_message,
        "ORIGINAL_MESSAGES": original_messages,
        "FEW_SHOT_ISSUES": fewshot_issues.model_dump(),
    })
    return run_agent(client, FEWSHOT_REWRITER_PROMPT, payload, MessagesOutput, model)


def rewrite_custom(
    client: OpenAI,
    original: str,
    requested_changes: str,
    model: str | None = None,
) -> DevRewriteOutput:
    payload = json.dumps({
        "ORIGINAL_DEVELOPER_MESSAGE": original,
        "REQUESTED_CHANGES": requested_changes,
    })
    return run_agent(client, CUSTOM_REWRITER_PROMPT, payload, DevRewriteOutput, model)
