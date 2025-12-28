from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, conf=0.5):
        """Instances of different YOLO models (COCO / LPR)"""
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Run the model on a single frame.

        Returns:
            detections: list of dicts:
              {'bbox': [x1,y1,x2,y2], 'conf': float, 'class_id': int, 'label': str}
        """
        results = self.model.predict(source=frame, conf=self.conf)  # returns Results object
        if len(results) == 0:
            return []

        res = results[0]
        boxes = res.boxes
        detections = []
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        # zip(xyxy, cls, conf)
        for box_xyxy, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box_xyxy.tolist())
            cid = int(cls.item()) if hasattr(cls, "item") else int(cls)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(conf.item() if hasattr(conf, "item") else conf),
                "class_id": cid,
                "label": self.model.names[cid] if cid in self.model.names else str(cid)
            })
        return detections
