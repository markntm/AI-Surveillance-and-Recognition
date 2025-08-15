from sqlalchemy import create_engine, Column, Integer, String, Enum, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import enum

Base = declarative_base()

class Recognition(enum.Enum):
    recognized = 1
    unrecognized = 2
    unsure = 0

class HumanBehavior(enum.Enum):
    walking = 1
    running = 2
    cycling = 3
    walking_pet = 4
    loitering = 5
    working = 6
    unknown = 0

class VehicleBehavior(enum.Enum):
    driving = 1
    speeding = 2
    stopped = 3
    parked = 4
    unknown = 0

class Event(Base):
    __tablename__ = 'events'
    event_id = Column(Integer, primary_key=True)
    object_id = Column(String)
    class_type = Column(String)
    time_first_detected = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime)
    recognition = Column(Enum(Recognition))

class Person(Base):
    __tablename__ = 'people'
    person_id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey('events.event_id'))
    appearance = Column(String)
    human_behavior = Column(Enum(HumanBehavior))

class Vehicle(Base):
    __tablename__ = 'vehicles'
    vehicle_id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey('events.event_id'))
    vehicle_type = Column(String)
    color = Column(String)
    license_plate = Column(String)
    vehicle_behavior = Column(Enum(VehicleBehavior))

engine = create_engine('sqlite:///surveillance.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def log_event(object_id, class_type):
    """Create event if new, else update last_seen"""
    event = session.query(Event).filter_by(object_id=object_id).first()
    now = datetime.datetime.utcnow()
    if not event:
        event = Event(object_id=object_id, class_type=class_type, last_seen=now)
        session.add(event)
        session.commit()
        return event
    else:
        event.last_seen = now
        session.commit()
        return event

def log_person(event, appearance, behavior):
    """Log/update person data linked to event"""
    person = session.query(Person).filter_by(event_id=event.event_id).first()
    if not person:
        person = Person(event_id=event.event_id, appearance=appearance, human_behavior=behavior)
        session.add(person)
    else:
        person.appearance = appearance
        person.human_behavior = behavior
    session.commit()

def log_vehicle(event, vehicle_type, color, license_plate, behavior):
    """Log/update vehicle data linked to event"""
    vehicle = session.query(Vehicle).filter_by(event_id=event.event_id).first()
    if not vehicle:
        vehicle = Vehicle(
            event_id=event.event_id,
            vehicle_type=vehicle_type,
            color=color,
            license_plate=license_plate,
            vehicle_behavior=behavior
        )
        session.add(vehicle)
    else:
        vehicle.vehicle_type = vehicle_type
        vehicle.color = color
        vehicle.license_plate = license_plate
        vehicle.vehicle_behavior = behavior
    session.commit()
