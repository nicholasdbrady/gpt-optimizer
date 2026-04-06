"""
FastAPI web application exposing the GPT prompt optimizer as an HTTP API.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from gpt_optimizer.models import OptimizeRequest, OptimizeResponse, PresetCheck
from gpt_optimizer.optimizer import optimize_from_request


# Load environment variables at startup
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    print("Starting GPT Optimizer API...")
    yield
    # Shutdown
    print("Shutting down GPT Optimizer API...")


# Create FastAPI app
app = FastAPI(
    title="GPT Optimizer API",
    description="API for optimizing GPT prompts",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/optimize")
async def api_optimize(request: OptimizeRequest):
    """
    Full prompt optimization endpoint.
    
    Accepts an OptimizeRequest and returns the optimization result.
    """
    try:
        result = optimize_from_request(request)
        return result.model_dump()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Optimization failed: {str(e)}"},
        )


@app.post("/api/check")
async def api_check(developer_message: str, check_type: str):
    """
    Targeted check endpoint.
    
    Args:
        developer_message: The prompt to check
        check_type: Type of check - one of "conflicts", "ambiguity", "output_format"
    
    Returns:
        OptimizeResponse with the check results
    """
    try:
        # Validate check_type
        valid_types = ["conflicts", "ambiguity", "output_format"]
        if check_type not in valid_types:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid check_type. Must be one of: {', '.join(valid_types)}"
                },
            )
        
        # Map check_type string to PresetCheck enum
        check_type_map = {
            "conflicts": PresetCheck.conflicting_instructions,
            "ambiguity": PresetCheck.ambiguity,
            "output_format": PresetCheck.output_format,
        }
        
        # Construct OptimizeRequest with the preset check
        request = OptimizeRequest(
            developer_message=developer_message,
            preset_check=check_type_map[check_type],
        )
        
        result = optimize_from_request(request)
        return result.model_dump()
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid request: {str(e)}"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Check failed: {str(e)}"},
        )


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("WEB_PORT", "8000"))
    
    # Run the application
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
