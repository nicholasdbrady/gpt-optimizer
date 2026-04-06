"""Configuration for the GPT Optimizer.

Supports three modes:
  1. Direct OpenAI — uses OPENAI_API_KEY for auth
  2. Azure OpenAI  — uses AZURE_OPENAI_ENDPOINT + DefaultAzureCredential
  3. Azure Foundry — uses AIProjectClient.get_openai_client() for auth
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.environ.get("OPTIMIZER_MODEL", "gpt-5.4")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Azure Foundry settings
AZURE_AI_PROJECT_ENDPOINT = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client using the best available auth method.

    Priority:
      1. Explicit api_key or OPENAI_API_KEY env var → standard OpenAI
      2. AZURE_OPENAI_ENDPOINT env var → Azure OpenAI with bearer token
      3. AZURE_AI_PROJECT_ENDPOINT env var → Foundry project client
    """
    key = api_key or OPENAI_API_KEY
    if key:
        return OpenAI(api_key=key)

    if AZURE_OPENAI_ENDPOINT:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise ImportError(
                "azure-identity is required for Azure OpenAI auth. "
                "Install with: pip install azure-identity"
            ) from exc
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        token = token_provider()
        return OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=token)

    if AZURE_AI_PROJECT_ENDPOINT:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.ai.projects import AIProjectClient
        except ImportError as exc:
            raise ImportError(
                "azure-ai-projects and azure-identity are required for Foundry auth. "
                "Install with: pip install azure-ai-projects azure-identity"
            ) from exc
        credential = DefaultAzureCredential()
        project_client = AIProjectClient(
            endpoint=AZURE_AI_PROJECT_ENDPOINT, credential=credential
        )
        return project_client.get_openai_client()

    raise ValueError(
        "No OpenAI credentials configured. Set OPENAI_API_KEY, "
        "AZURE_OPENAI_ENDPOINT, or AZURE_AI_PROJECT_ENDPOINT."
    )
