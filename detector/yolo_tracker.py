from ultralytics import YOLO
import time


class YOLOTracker:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_data = {}

    def track_persons(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            classes=[0],  # PERSON ONLY
            tracker="bytetrack.yaml"
        )

        persons = []
        current_time = time.time()

        if results[0].boxes.id is None:
            return persons

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, pid in zip(boxes, ids):
            pid = int(pid)
            x1, y1, x2, y2 = map(int, box)
            cy = (y1 + y2) // 2

            if pid not in self.person_data:
                self.person_data[pid] = {
                    "start": current_time,
                    "last_y": cy,
                    "dir": "STAY"
                }

            pdata = self.person_data[pid]

            if cy < pdata["last_y"]:
                pdata["dir"] = "UP"
            elif cy > pdata["last_y"]:
                pdata["dir"] = "DOWN"

            pdata["last_y"] = cy
            elapsed = int(current_time - pdata["start"])

            persons.append({
                "id": pid,
                "box": (x1, y1, x2, y2),
                "time": elapsed,
                "direction": pdata["dir"]
            })

        return persons