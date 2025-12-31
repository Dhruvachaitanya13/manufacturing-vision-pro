from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

from .config import settings
from .inference import InferenceEngine
from .schemas import InspectionResponse, HealthCheckResponse

# Initialize API
app = FastAPI(
    title="🏭 Manufacturing Vision Pro API",
    description="Enterprise-grade visual inspection microservice.",
    version=settings.VERSION,
    debug=settings.DEBUG_MODE
)

# CORS (Security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine Instance
engine = InferenceEngine()

@app.get("/", tags=["System"])
def root():
    return {
        "service": "Manufacturing Vision Pro",
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check():
    """Returns system operational status."""
    is_ready = engine.model is not None
    return {
        "status": "operational" if is_ready else "degraded",
        "device": settings.DEVICE,
        "model_loaded": is_ready
    }

@app.post("/inspect", response_model=InspectionResponse, tags=["Inference"])
async def inspect_part(file: UploadFile = File(...)):
    """
    Main entry point for visual inspection.
    Accepts image file -> Returns defect coordinates and PASS/FAIL status.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Only JPEG/PNG images are supported."
        )

    try:
        content = await file.read()
        response = engine.predict(content, file.filename)
        return response
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail="Model service unavailable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Processing Error")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)