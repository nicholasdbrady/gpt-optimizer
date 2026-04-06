"""CLI entry point for gpt-optimizer.

Usage:
    python -m gpt_optimizer "Your prompt here"
    python -m gpt_optimizer --mode pro "Your prompt"
    python -m gpt_optimizer --check conflicts "Your prompt"
    python -m gpt_optimizer --changes "Add error handling" "Your prompt"
"""

import argparse
import json
import sys

from .models import OptimizerMode, PresetCheck
from .optimizer import optimize_prompt


def main():
    parser = argparse.ArgumentParser(
        prog="gpt-optimizer",
        description="Optimize prompts using OpenAI's multi-agent approach",
    )
    parser.add_argument("prompt", help="The developer/system prompt to optimize")
    parser.add_argument(
        "--mode",
        choices=["instant", "default", "pro"],
        default="default",
        help="Optimization depth (default: default)",
    )
    parser.add_argument(
        "--check",
        choices=["conflicts", "ambiguity", "output_format"],
        help="Run a targeted check only",
    )
    parser.add_argument(
        "--changes",
        help="Free-text custom optimization instruction",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the agent model (default: gpt-5.4)",
    )
    parser.add_argument(
        "--target",
        default="gpt-5.4",
        help="Target model to optimize for (default: gpt-5.4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON instead of formatted text",
    )

    args = parser.parse_args()

    preset_map = {
        "conflicts": PresetCheck.conflicting_instructions,
        "ambiguity": PresetCheck.ambiguity,
        "output_format": PresetCheck.output_format,
    }

    result = optimize_prompt(
        developer_message=args.prompt,
        mode=OptimizerMode(args.mode),
        preset_check=preset_map.get(args.check) if args.check else None,
        requested_changes=args.changes,
        model=args.model,
        target_model=args.target,
    )

    if args.output_json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(f"\n{'─' * 60}")
        print(f"  {result.summary}")
        print(f"{'─' * 60}\n")

        if result.comments:
            print("Changes:")
            for c in result.comments:
                icon = "⚠️" if c.kind == "finding" else "✏️"
                print(f"  {icon}  {c.reason}")
            print()

        print("Optimized prompt:")
        print(f"{'─' * 40}")
        print(result.new_developer_message)
        print(f"{'─' * 40}\n")


if __name__ == "__main__":
    main()
