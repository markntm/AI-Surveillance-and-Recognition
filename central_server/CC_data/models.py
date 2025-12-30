from sqlalchemy import Column, Integer, Float, String, Enum, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from database import Base


class Recognition(enum.Enum):
    unknown = 0
    recognised = 1
    unrecognised = 2


class Behavior(enum.Enum):
    unknown = 0
    moving = 1
    speeding = 2
    stopped = 3


class VehicleFunction(enum.Enum):
    unknown = 0
    personal = 1
    utility = 2
    first_responder = 3
    delivery = 4


class Event(Base):
    """Entre for when something important happens"""
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)

    camera_id = Column(String, nullable=False)  # name of street or location of surveillance
    timestamp = Column(DateTime, default=datetime.now(), nullable=False)
    threat_score = Column(Integer, default=0)
    image_path = Column(String, default=None)

    objects = relationship("DetectedObject", back_populates="event")


class DetectedObject(Base):
    """Each object detected in the frame of an event"""
    __tablename__ = 'detected_objects'
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)

    object_type = Column(String, nullable=False)  # YOLO label of the detected object (e.g., person)
    behavior = Column(Enum(Behavior), nullable=False)
    recognition = Column(Enum(Recognition), nullable=False)
    confidence = Column(Float, nullable=False)

    event = relationship("Event", back_populates="objects")
    vehicle = relationship("Vehicle", uselist=False, back_populates="object")


class Vehicle(Base):
    """Additional columns for when vehicles are detected"""
    __tablename__ = 'vehicles'
    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('detected_objects.id'), nullable=False)

    license_plate = Column(String, nullable=False)
    primary_color = Column(String, nullable=False)
    vehicle_type = Column(String, nullable=False)
    vehicle_function = Column(Enum(VehicleFunction), nullable=False)

    object = relationship("DetectedObject", back_populates="vehicle")
