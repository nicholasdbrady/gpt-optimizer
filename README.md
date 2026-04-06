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
result = asyncio.run(optimize_prompt(
    developer_message="Your prompt here",
    preset_check=PresetCheck.conflicting_instructions,
))

# Custom changes
result = asyncio.run(optimize_prompt(
    developer_message="Your prompt here",
    requested_changes="Add error handling instructions",
))
```

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
| `AZURE_OPENAI_ENDPOINT` | `https://swdn-resource.openai.azure.com/openai/v1/` | Azure OpenAI endpoint |
| `AZURE_AI_PROJECT_ENDPOINT` | `https://swdn-resource.services.ai.azure.com/api/projects/foundry-project` | Foundry project endpoint |
| `OPTIMIZER_MODEL` | `gpt-5.4` | Model used for all checker/rewriter agents |

**Authentication**: Run `az login` for Microsoft Entra ID (default). Pass `--api-key <key>` to the CLI for direct API key auth.

## Background

Reverse-engineered from OpenAI's `POST /v1/dashapi/optimize/promptv2` endpoint and the agent prompts published in the [OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Optimize_Prompts.ipynb). This project provides the same multi-agent optimization pipeline as a standalone library, CLI, and API.

## License

MIT
