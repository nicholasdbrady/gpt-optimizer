#!/usr/bin/env bash
set -euo pipefail

# GPT Prompt Optimizer wrapper script
#
# Usage:
#   ./scripts/optimize.sh "Your prompt here"
#   ./scripts/optimize.sh "Your prompt" --mode pro
#   ./scripts/optimize.sh "Your prompt" --check conflicts
#   ./scripts/optimize.sh "Your prompt" --changes "make it concise"
#
# The script auto-adds --json, so the caller always gets structured output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check auth: Azure CLI (Entra ID) is the default
if ! az account show &>/dev/null; then
    echo "Error: Not authenticated. Run 'az login' first." >&2
    echo "  (Or pass --api-key <key> for direct API key auth)" >&2
    exit 1
fi

# Check python is available
if ! command -v python &>/dev/null; then
    echo "Error: python not found. Install Python 3.10+ first." >&2
    exit 1
fi

# Check package is installed, install if missing
if ! python -c "import gpt_optimizer" &>/dev/null; then
    echo "Installing gpt-optimizer..." >&2
    pip install -e "$PROJECT_DIR" --quiet
fi

# Add --json if not already present
args=("$@")
if [[ ! " ${args[*]} " =~ " --json " ]]; then
    args+=("--json")
fi

cd "$PROJECT_DIR"
exec python -m gpt_optimizer "${args[@]}"
