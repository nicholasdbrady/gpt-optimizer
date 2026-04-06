#!/usr/bin/env python3
"""
Foundry Agent Deployment Script for GPT Prompt Optimizer

Sets up the prompt optimizer as a Microsoft Foundry agent using the azure-ai-projects SDK.
Requires AZURE_AI_PROJECT_ENDPOINT and optionally FOUNDRY_MODEL_DEPLOYMENT_NAME env vars.

Usage:
    python foundry_setup.py
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import PromptAgentDefinition, FunctionTool
    from azure.identity import DefaultAzureCredential
except ImportError as e:
    print(f"Error: Required Azure SDK not installed. Install with:")
    print("  pip install azure-ai-projects azure-identity python-dotenv")
    sys.exit(1)


def create_optimize_prompt_tool() -> FunctionTool:
    """
    Create a FunctionTool that wraps the optimize_prompt function.
    
    Returns:
        FunctionTool: Configured function tool for the agent
    """
    return FunctionTool(
        function={
            "name": "optimize_prompt",
            "description": "Optimize a developer or system prompt for clarity, consistency, and format compliance",
            "parameters": {
                "type": "object",
                "properties": {
                    "developer_message": {
                        "type": "string",
                        "description": "The prompt text to optimize"
                    },
                    "optimizer_mode": {
                        "type": "string",
                        "enum": ["instant", "default", "pro"],
                        "description": "Optimization speed/depth tradeoff. instant=quick feedback, default=balanced, pro=deep analysis"
                    },
                    "preset_check": {
                        "type": "string",
                        "enum": ["conflicting_instructions", "ambiguity", "output_format"],
                        "description": "Optional targeted consistency check"
                    },
                    "requested_changes": {
                        "type": "string",
                        "description": "Custom optimization instruction for specific requirements"
                    }
                },
                "required": ["developer_message"]
            }
        }
    )


def setup_foundry_agent() -> None:
    """
    Set up and deploy the prompt optimizer as a Foundry agent.
    
    This function:
    1. Authenticates with Azure using DefaultAzureCredential
    2. Creates an AIProjectClient
    3. Defines a PromptAgentDefinition with model, instructions, and tools
    4. Creates or updates the agent version
    5. Prints deployment details
    
    Raises:
        ValueError: If required environment variables are missing
        Exception: If agent creation/update fails
    """
    # Get environment variables
    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    model_deployment = os.getenv("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-5-mini")
    
    if not project_endpoint:
        raise ValueError(
            "AZURE_AI_PROJECT_ENDPOINT environment variable is required. "
            "Set it in your .env file or system environment."
        )
    
    print(f"📦 Setting up Foundry Agent Deployment")
    print(f"   Project endpoint: {project_endpoint}")
    print(f"   Model deployment: {model_deployment}")
    print()
    
    # Authenticate with Azure
    try:
        print("🔐 Authenticating with Azure...")
        credential = DefaultAzureCredential()
        print("   ✓ Authentication successful")
    except Exception as e:
        print(f"   ✗ Authentication failed: {e}")
        raise
    
    # Create AIProjectClient
    try:
        print("🚀 Creating AIProjectClient...")
        client = AIProjectClient.from_config(
            credential=credential,
            endpoint=project_endpoint
        )
        print("   ✓ Client created")
    except Exception as e:
        print(f"   ✗ Client creation failed: {e}")
        raise
    
    # Create tools
    print("🔧 Configuring agent tools...")
    tools = [create_optimize_prompt_tool()]
    print(f"   ✓ {len(tools)} tool(s) configured")
    
    # Define agent
    print("📋 Defining PromptAgentDefinition...")
    agent_definition = PromptAgentDefinition(
        model=model_deployment,
        instructions=(
            "You are an expert prompt engineer and optimization specialist. "
            "Your role is to help users improve their prompts for clarity, consistency, and format compliance. "
            "Analyze prompts for potential ambiguities, conflicting instructions, and formatting issues. "
            "Provide actionable suggestions to enhance prompt quality and effectiveness. "
            "Focus on improving user intent clarity, reducing hallucination, and ensuring structured outputs."
        ),
        tools=tools
    )
    print("   ✓ Agent definition created")
    
    # Deploy agent
    print("🌐 Deploying agent to Foundry...")
    try:
        agent = client.agents.create_version(agent_definition)
        print("   ✓ Agent deployed successfully")
        print()
        print("=" * 60)
        print("✅ Deployment Complete")
        print("=" * 60)
        print(f"Agent Name:    {agent.name if hasattr(agent, 'name') else 'prompt-optimizer'}")
        print(f"Agent ID:      {agent.id if hasattr(agent, 'id') else 'N/A'}")
        print(f"Model:         {model_deployment}")
        print(f"Tools:         optimize_prompt")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"   ✗ Agent deployment failed: {e}")
        raise


def main() -> int:
    """
    Main entry point for the deployment script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        setup_foundry_agent()
        return 0
    except ValueError as e:
        print(f"❌ Configuration Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Deployment Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
