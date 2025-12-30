from fastapi import FastAPI, Depends, Header, HTTPException
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os

from central_server.CC_data.database import engine, Base, SessionLocal
from central_server.CC_data.event_log import *
from central_server.CC_data.models import Behavior, Recognition, VehicleFunction
from central_server.CC_data.schemas import EventIn
from secret import dev_key, allowed_IP


app = FastAPI(title="Surveillance Dashboard", version="0.1")
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_IP],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

API_KEY = os.getenv("SURVEILLANCE_API_KEY", dev_key)


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/api/events", dependencies=[Depends(verify_api_key)])
def ingest_event(event_in: EventIn, db: Session = Depends(get_db)):
    # 1. Create Event
    event = create_event(
        db=db,
        camera_id=event_in.camera_id,
        threat_score=event_in.threat_score,
        image_path=event_in.image_path
    )

    # 2. Add detected objects
    for obj in event_in.objects:
        detected = add_detected_object(
            db=db,
            event_id=event.id,
            object_type=obj.object_type,
            behavior=Behavior[obj.behavior],
            recognition=Recognition[obj.recognition],
            confidence=obj.confidence
        )

        # 3. Optional vehicle
        if obj.vehicle:
            add_vehicle(
                db=db,
                object_id=detected.id,
                license_plate=obj.vehicle.license_plate,
                primary_color=obj.vehicle.primary_color,
                vehicle_type=obj.vehicle.vehicle_type,
                vehicle_function=VehicleFunction[obj.vehicle.vehicle_function]
            )

    return {"status": "ok", "event_id": event.id}
