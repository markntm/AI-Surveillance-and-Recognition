from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
import os

from CC_data.database import engine, Base, SessionLocal
from CC_data.event_log import *
from CC_data.models import Behavior, Recognition, VehicleFunction, Event
from CC_data.schemas import EventIn
from secret import dev_key, allowed_IP


app = FastAPI(title="Surveillance Dashboard", version="0.1")
app.mount("/static", StaticFiles(directory="CC_dashboard/static"), name="static")
templates = Jinja2Templates(directory="CC_dashboard/templates")
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


@app.get("/", include_in_schema=False)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/status")
def get_status(db: Session = Depends(get_db)):
    """
    High-level system status for dashboard health checks.
    """

    last_event = (
        db.query(Event)
        .order_by(Event.timestamp.desc())
        .first()
    )

    total_events = db.query(Event).count()

    return {
        "server": "online",
        "total_events": total_events,
        "last_event_time": last_event.timestamp.isoformat() if last_event else None
    }


@app.get("/api/events")
def get_events(
        limit: int = 10,
        offset: int = 0,
        camera_id: Optional[str] = None,
        min_threat: Optional[int] = None,
        db: Session = Depends(get_db)
):
    """
    Retrieve recent surveillance events with detected objects and vehicles.
    """

    query = db.query(Event)

    # ---- Filtering ----
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)

    if min_threat is not None:
        query = query.filter(Event.threat_score >= min_threat)

    total_count = query.count()

    events = (
        query
        .order_by(Event.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    response = []

    for event in events:
        event_data = {
            "id": event.id,
            "camera_id": event.camera_id,
            "timestamp": event.timestamp.isoformat(),
            "threat_score": event.threat_score,
            "image_path": event.image_path,
            "objects": []
        }

        for obj in event.objects:
            obj_data = {
                "object_type": obj.object_type,
                "behavior": obj.behavior.name,
                "recognition": obj.recognition.name,
                "confidence": obj.confidence,
                "vehicle": None
            }

            if obj.vehicle:
                obj_data["vehicle"] = {
                    "license_plate": obj.vehicle.license_plate,
                    "primary_color": obj.vehicle.primary_color,
                    "vehicle_type": obj.vehicle.vehicle_type,
                    "vehicle_function": obj.vehicle.vehicle_function.name
                }

            event_data["objects"].append(obj_data)

        response.append(event_data)

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "events": response
    }


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
