"""
Health check endpoints
"""

from fastapi import APIRouter
from models.schemas import HealthResponse
from services.model_loader import model_loader
import time

router = APIRouter()

startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=len(model_loader.loaded_models),
        uptime_seconds=time.time() - startup_time
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for k8s/Cloud Run"""
    return {
        "status": "ready",
        "models_loaded": len(model_loader.loaded_models)
    }

