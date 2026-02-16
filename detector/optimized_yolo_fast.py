from ultralytics import YOLO
import torch
import time
import numpy as np

class OptimizedYOLOFast:
    def __init__(self, model_path, confidence):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()  # FP16 for speed
            self.model.fuse()  # fuse conv + bn
        self.confidence = confidence
        self.person_data = {}

    def process(self, frame):
        # Frame is already resized by VideoStream to 640x360
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=[0, 39, 41, 56, 60, 63],  # Person + objects
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device
        )

        persons, objects = [], []
        current_time = time.time()
        r = results[0]

        if r.boxes.id is None:
            return persons, objects

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            label = self.model.names[int(cls)]

            if cls == 0:  # PERSON
                obj_id = int(obj_id)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Initialize if new
                if obj_id not in self.person_data:
                    self.person_data[obj_id] = {
                        "start": current_time,
                        "last_x": cx,
                        "last_y": cy,
                        "dir": "STAY"
                    }

                pdata = self.person_data[obj_id]

                # Determine direction
                dy = cy - pdata["last_y"]
                dx = cx - pdata["last_x"]
                move_threshold = 5  # pixels to consider movement

                if abs(dx) > move_threshold or abs(dy) > move_threshold:
                    pdata["dir"] = "MOVING"
                elif dy < 0:
                    pdata["dir"] = "UP"
                elif dy > 0:
                    pdata["dir"] = "DOWN"
                else:
                    pdata["dir"] = "STAY"

                pdata["last_x"] = cx
                pdata["last_y"] = cy
                elapsed = int(current_time - pdata["start"])

                persons.append({
                    "id": obj_id,
                    "box": (x1, y1, x2, y2),
                    "time": elapsed,
                    "direction": pdata["dir"]
                })
            else:
                objects.append({"box": (x1, y1, x2, y2), "label": label})

        return persons, objects
