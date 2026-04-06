# GPT Prompt Optimizer

Programmatic prompt optimization using the same multi-agent approach as OpenAI's dashboard optimizer. Uses gpt-5.4 via Azure OpenAI with Microsoft Entra ID authentication.

## Quick Start

```bash
pip install -e .
az login                          # Microsoft Entra ID authentication

python -m gpt_optimizer "You are a helpful assistant." --json
```

## Usage

### Python Library

`optimize_prompt` returns an `OptimizeResponse`:

```python
from gpt_optimizer import optimize_prompt, OptimizerMode, PresetCheck

# Basic optimization (uses az login credentials)
result = optimize_prompt(
    developer_message="Your prompt here",
    mode=OptimizerMode.default,
)
print(result.new_developer_message)

# Targeted check
result = optimize_prompt(
    developer_message="Your prompt here",
    preset_check=PresetCheck.conflicting_instructions,
)

# Custom changes
result = optimize_prompt(
    developer_message="Your prompt here",
    requested_changes="Add error handling instructions",
)
```

### End-to-End Examples

These walkthroughs show real input and output captured from live runs against Azure OpenAI gpt-5.4.

<details>
<summary><strong>Example 1: Detecting and resolving contradictions</strong></summary>

A prompt that says "always return JSON" but also "respond in plain text only":

**Python:**
```python
from gpt_optimizer import optimize_prompt

result = optimize_prompt(
    "You are a data extraction assistant. Always return results as valid JSON objects. "
    "Do not use any structured or formatted output — respond in plain conversational "
    "text only. Include all extracted fields in your response."
)
```

**CLI:**
```bash
python -m gpt_optimizer "You are a data extraction assistant. Always return results as valid JSON objects. Do not use any structured or formatted output — respond in plain conversational text only. Include all extracted fields in your response."
```

**Output:**
```
────────────────────────────────────────────────────────────
  Applied a full optimize pass with 7 change(s).
────────────────────────────────────────────────────────────

Changes:
  ⚠️  Contradiction: The prompt says to "Always return results as valid JSON
     objects" but also says "Do not use any structured or formatted output —
     respond in plain conversational text only"; valid JSON is structured
     output, so both cannot be followed simultaneously.
  ⚠️  Format: No JSON schema or required field list is specified.
  ⚠️  Format: Data types for the "extracted fields" are unspecified.
  ⚠️  Format: Error-handling is not specified for missing fields.
  ✏️  Rewrote developer prompt to resolve detected issues.
```

**Optimized prompt:**
```
You are a data extraction assistant. Always return results as valid JSON
objects. Include all extracted fields in your response.

## Output Format
Return a single valid JSON object containing all extracted fields.
- Use JSON key-value pairs for each extracted field.
- If a field is missing, ambiguous, or cannot be extracted, use null.
- Return only the JSON object and no additional text.
```

**What happened:** The optimizer detected the contradiction between "JSON objects" and "plain text only", removed the conflicting clause, and added a concrete `## Output Format` section with field-level guidance.

</details>

<details>
<summary><strong>Example 2: Applying custom changes</strong></summary>

Specialize a generic support prompt for billing disputes:

**Python:**
```python
from gpt_optimizer import optimize_prompt

result = optimize_prompt(
    "You are a customer support agent. Help users with their questions. "
    "Be professional and empathetic. Always acknowledge the customer's "
    "concern before providing a solution.",
    requested_changes="specialize this prompt for handling billing disputes and refund requests"
)
```

**CLI:**
```bash
python -m gpt_optimizer "You are a customer support agent. Help users with their questions. Be professional and empathetic. Always acknowledge the customer's concern before providing a solution." --changes "specialize this prompt for handling billing disputes and refund requests"
```

**JSON output** (`--json`):
```json
{
  "new_developer_message": "You are a customer support agent specializing in billing disputes and refund requests. Help users with their questions about charges, billing issues, and refunds. Be professional and empathetic. Always acknowledge the customer's concern before providing a solution.",
  "summary": "Applied custom optimization: specialize this prompt for handling billing disputes and refund requests",
  "comments": [
    {
      "kind": "explanation",
      "reason": "Applied requested changes: specialize this prompt for handling billing disputes and refund requests"
    }
  ],
  "operation_mode": "custom"
}
```

**What happened:** The optimizer added "specializing in billing disputes and refund requests" and "about charges, billing issues, and refunds" while preserving the original professional/empathetic tone.

</details>

<details>
<summary><strong>Example 3: Targeted conflict check</strong></summary>

Find the word-count contradiction without doing a full optimization:

**Python:**
```python
from gpt_optimizer import optimize_prompt, PresetCheck

result = optimize_prompt(
    "You are a research summarizer. Keep all answers under 50 words. "
    "Provide thorough, detailed explanations of at least 500 words for every topic.",
    preset_check=PresetCheck.conflicting_instructions,
)
print(result.issues_found)  # True
print(result.preset_check)  # "conflicting_instructions"
```

**CLI:**
```bash
python -m gpt_optimizer "You are a research summarizer. Keep all answers under 50 words. Provide thorough, detailed explanations of at least 500 words for every topic." --check conflicts --json
```

**JSON output:**
```json
{
  "issues_found": true,
  "operation_mode": "preset_check",
  "preset_check": "conflicting_instructions",
  "comments": [
    {
      "kind": "finding",
      "reason": "\"Keep all answers under 50 words\" contradicts \"Provide thorough, detailed explanations of at least 500 words for every topic,\" since a response cannot be both under 50 words and at least 500 words."
    },
    {
      "kind": "explanation",
      "reason": "Rewrote prompt to resolve 1 issue(s)."
    }
  ],
  "new_developer_message": "You are a research summarizer. Provide thorough, detailed explanations of at least 500 words for every topic.",
  "summary": "conflicting_instructions check found 1 issue(s)."
}
```

**What happened:** The targeted check found the 50-word vs 500-word conflict and resolved it by keeping the 500-word clause (which better preserves the "thorough" intent).

</details>

### CLI

```bash
python -m gpt_optimizer "Your prompt"                          # default mode
python -m gpt_optimizer "Your prompt" --mode pro               # pro mode (second pass)
python -m gpt_optimizer "Your prompt" --check conflicts        # targeted check
python -m gpt_optimizer "Your prompt" --check ambiguity
python -m gpt_optimizer "Your prompt" --check output_format
python -m gpt_optimizer "Your prompt" --changes "Add error handling"
python -m gpt_optimizer "Your prompt" --model gpt-5-mini       # override agent model
python -m gpt_optimizer "Your prompt" --target gpt-5-mini      # optimize for target model
python -m gpt_optimizer "Your prompt" --json                   # raw JSON output
```

| Flag | Description |
|------|-------------|
| `--mode {instant,default,pro}` | Optimization depth (default: `default`) |
| `--check {conflicts,ambiguity,output_format}` | Run a single targeted check |
| `--changes TEXT` | Free-text custom optimization instruction |
| `--model MODEL` | Override agent model (default: gpt-5.4) |
| `--target MODEL` | Target model to optimize for (default: gpt-5.4) |
| `--json` | Output raw JSON instead of formatted text |

### Web API

```bash
pip install -e ".[web]"
python web_app.py              # default port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/api/optimize` | Full prompt optimization |
| `POST` | `/api/check` | Targeted check (query params: `developer_message`, `check_type`) |

**POST /api/optimize** — request body:

```json
{
  "developer_message": "Your prompt here",
  "optimizer_mode": "default",
  "model_name": "gpt-5.4",
  "optimizing_for": "gpt-5.4",
  "preset_check": null,
  "requested_changes": null,
  "messages": [{"role": "user", "content": ""}],
  "tools": []
}
```

**Response:**

```json
{
  "new_developer_message": "...",
  "summary": "Applied a full optimize pass with 3 change(s).",
  "comments": [
    {"kind": "finding", "reason": "...", "location": null}
  ],
  "issues_found": true,
  "new_messages": [],
  "operation_mode": "full_optimize",
  "preset_check": null
}
```

### Foundry Agent

```bash
pip install -e ".[foundry]"
python foundry_setup.py
```

## Architecture

Five specialized agents running on gpt-5.4, orchestrated in a parallel-check → conditional-rewrite pipeline:

```
                        Input Prompt
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     Contradiction      Format        Few-Shot
       Checker          Checker        Checker
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                       Dev Rewriter
                             │
                             ▼
                     Few-Shot Rewriter
                      (if examples)
                             │
                             ▼
                      Optimized Prompt
```

- **instant** — single rewrite pass, no checkers
- **default** — parallel checkers → conditional rewrite
- **pro** — default + a second check-and-rewrite pass on the output

## Configuration

Set via environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint (required) |
| `AZURE_AI_PROJECT_ENDPOINT` | — | Foundry project endpoint |
| `OPTIMIZER_MODEL` | `gpt-5.4` | Model used for all checker/rewriter agents |

**Authentication**: Run `az login` for Microsoft Entra ID (default). Pass `--api-key <key>` to the CLI for direct API key auth.

## Testing

### Unit tests (mocked, no API calls)

```bash
pytest -m "not live"           # 102 tests, ~1s
```

### Live integration tests (real Azure OpenAI calls)

```bash
az login                       # authenticate first
pytest -m live -v              # 12 tests, ~90s
```

The live tests synthetically generate prompts with known defects and verify the optimizer correctly detects and fixes them:

| Test | Prompt | What it verifies |
|------|--------|-----------------|
| **Contradiction detection** | `"Always return results as valid JSON objects. Do not use any structured output — respond in plain text only."` | Detects the JSON vs plain text contradiction and resolves it |
| **Clean prompt passthrough** | `"You are a Python coding assistant. Help users write clean, efficient code."` | No issues found, core intent preserved |
| **Format ambiguity** | `"Extract key information and return results in a structured format."` | Flags the vague "structured format" and adds a concrete schema |
| **Custom changes** | `"You are a customer support agent."` + changes: `"specialize for billing disputes"` | Adds billing/refund language while preserving professional tone |
| **Preset conflict check** | `"Keep answers under 50 words. Provide detailed explanations of at least 500 words."` | Targeted `conflicting_instructions` check finds the word count conflict |
| **Few-shot inconsistency** | System: `"Always respond with exactly one sentence."` Example: 4-sentence response | Detects the multi-sentence example violating the one-sentence rule |

### Run everything

```bash
pytest -v                      # all 114 tests (102 mocked + 12 live)
```

## Background

Reverse-engineered from OpenAI's `POST /v1/dashapi/optimize/promptv2` endpoint and the agent prompts published in the [OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Optimize_Prompts.ipynb). This project provides the same multi-agent optimization pipeline as a standalone library, CLI, and API.

## License

MIT
