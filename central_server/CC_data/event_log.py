

def log_event(object_id, class_id):
    """Create event if new, else update last_seen"""
    event = session.query(Event).filter_by(object_id=object_id).first()
    now = datetime.datetime.now(datetime.UTC)
    if not event:
        event = Event(
            object_id=object_id,
            class_id=class_id,
            last_seen=now
        )
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
        person = Person(
            event_id=event.event_id,
            appearance=appearance,
            human_behavior=behavior
        )
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
