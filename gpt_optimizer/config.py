"""Configuration for the GPT Optimizer.

Authentication priority:
  1. Microsoft Entra ID via DefaultAzureCredential (az login) — default
  2. Explicit api_key parameter — optional fallback
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.environ.get("OPTIMIZER_MODEL", "gpt-5.4")

AZURE_OPENAI_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://swdn-resource.openai.azure.com/openai/v1/",
)
AZURE_AI_PROJECT_ENDPOINT = os.environ.get(
    "AZURE_AI_PROJECT_ENDPOINT",
    "https://swdn-resource.services.ai.azure.com/api/projects/foundry-project",
)


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client for Azure OpenAI.

    Auth priority:
      1. Explicit api_key → used directly with Azure OpenAI endpoint
      2. DefaultAzureCredential (az login) → Entra ID bearer token

    Requires AZURE_OPENAI_ENDPOINT to be set (defaults to swdn-resource).
    """
    if api_key:
        return OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=api_key)

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=token_provider)
