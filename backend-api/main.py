"""
Cognirace Real-Time Prediction API
FastAPI service for ML model predictions on port 8005
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from config.settings import settings
from models.schemas import HealthResponse, ErrorResponse
import time
import sys
import traceback

# Startup time
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    print("="*70)
    print("🏁 COGNIRACE PREDICTION API")
    print("="*70)
    print(f"Port: {settings.api_port}")
    print(f"Project: {settings.gcp_project_id}")
    print(f"Models bucket: gs://{settings.gcs_bucket_models}")
    print("="*70)
    
    yield
    
    # Shutdown
    print("\n✓ API shutdown")


# Initialize FastAPI app
app = FastAPI(
    title="Cognirace Prediction API",
    description="Real-time ML predictions for GR Cup race strategy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from routers import predict, health

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/predict", tags=["predictions"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"✗ Error: {exc}")
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Cognirace Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predictions": "/predict/*"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Disable reload for production
        log_level=settings.api_log_level,
        workers=1
    )
