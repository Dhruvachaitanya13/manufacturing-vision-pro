from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class DefectDetail(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox

class InspectionResponse(BaseModel):
    filename: str
    status: str = Field(..., description="Decision: PASS or REJECT")
    inference_time_ms: float
    count: int
    detections: List[DefectDetail]

class HealthCheckResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool