from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from models import Event, DetectedObject, Vehicle, Recognition, Behavior, VehicleFunction


# ---------- Write ----------

def create_event(
        db: Session,
        camera_id: str,
        threat_score: int,
        image_path: Optional[str] = None,
        timestamp: Optional[datetime] = None
) -> Event:

    """
    Create a new Event row.
    """

    event = Event(
        camera_id=camera_id,
        threat_score=threat_score,
        image_path=image_path,
        timestamp=timestamp or datetime.utcnow()
    )

    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def add_detected_object(
        db: Session,
        event_id: int,
        object_type: str,
        behavior: Behavior,
        recognition: Recognition,
        confidence: float
) -> DetectedObject:

    """
    Add a detected object to an existing event.
    """

    obj = DetectedObject(
        event_id=event_id,
        object_type=object_type,
        behavior=behavior,
        recognition=recognition,
        confidence=confidence
    )

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def add_vehicle(
        db: Session,
        object_id: int,
        license_plate: str,
        primary_color: str,
        vehicle_type: str,
        vehicle_function: VehicleFunction
) -> Vehicle:

    """
    Attach vehicle-specific metadata to a detected object.
    """

    vehicle = Vehicle(
        object_id=object_id,
        license_plate=license_plate,
        primary_color=primary_color,
        vehicle_type=vehicle_type,
        vehicle_function=vehicle_function
    )

    db.add(vehicle)
    db.commit()
    db.refresh(vehicle)
    return vehicle


# ---------- Read ----------

def get_event_by_id(db: Session, event_id: int) -> Optional[Event]:
    return db.query(Event).filter(Event.id == event_id).first()


def get_recent_events(db: Session, limit: int = 100) -> List[Event]:
    return (db.query(Event).order_by(Event.timestamp.desc()).limit(limit).all())


def get_events_by_camera(db: Session, camera_id: str, limit: int = 100) -> List[Event]:
    return (db.query(Event).filter(Event.camera_id == camera_id).order_by(Event.timestamp.desc()).limit(limit).all())


# ---------- Delete ----------

def delete_event(db: Session, event_id: int) -> bool:
    event = get_event_by_id(db, event_id)
    if not event:
        return False

    db.delete(event)
    db.commit()
    return True
