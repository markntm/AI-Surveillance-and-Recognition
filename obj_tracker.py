from deep_sort_realtime.deepsort_tracker import DeepSort

# not used

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3)

    def update(self, detections, frame):
        """
        detections: list of dicts as returned by ObjectDetector.detect
        frame: numpy array for appearance embedding (DeepSORT uses it)

        Returns: list of track dicts:
          {'track_id': int, 'bbox': [x1,y1,x2,y2], 'label': label, 'conf': conf}
        """
        ds_input = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], d["conf"], d["class_id"]))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            try:
                ltrb = t.to_ltrb()  # left, top, right, bottom
            except Exception:
                continue
            # t.get_det_class() / t.get_det_conf() available via deep-sort wrapper
            label = t.get_det_class() if hasattr(t, "get_det_class") else None
            conf = t.get_det_conf() if hasattr(t, "get_det_conf") else None
            out.append({
                "track_id": t.track_id,
                "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                "label": label,
                "conf": conf
            })
        return out
