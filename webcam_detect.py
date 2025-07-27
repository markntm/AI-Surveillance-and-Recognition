from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load pre-trained YOLOv8 model
model = YOLO("yolo11n.pt")  # (n/s/m/l/x)

# Initializes DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model.predict(source=frame, save=False, conf=0.4)
    detections = results[0].boxes

    # Convert detections to DeepSORT input format
    dets_for_deepsort = []
    if detections is not None and detections.xyxy is not None:
        for box, cls, conf in zip(detections.xyxy, detections.cls, detections.conf):
            x1, y1, x2, y2 = box.tolist()
            dets_for_deepsort.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), int(cls.item())))

    # Run DeepSORT tracker
    tracks = tracker.update_tracks(dets_for_deepsort, frame=frame)

    # Annotate tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Retrieve custom attributes (we'll store class and confidence here)
        class_id = track.get_det_class()
        confidence = track.get_det_conf()
        class_name = model.names[int(class_id)] if class_id is not None else "Unknown"
        conf_str = f"{confidence:.2f}" if confidence is not None else "?.??"

        label = f"ID {track_id} | {class_name} ({conf_str})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8 + DeepSORT", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
