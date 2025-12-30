from pydantic import BaseModel
from typing import List, Optional


class VehicleIn(BaseModel):
    license_plate: str
    primary_color: str
    vehicle_type: str
    vehicle_function: str


class DetectedObjectIn(BaseModel):
    object_type: str
    behavior: str
    recognition: str
    confidence: float
    vehicle: Optional[VehicleIn] = None


class EventIn(BaseModel):
    camera_id: str
    threat_score: int
    image_path: Optional[str] = None
    objects: List[DetectedObjectIn]
