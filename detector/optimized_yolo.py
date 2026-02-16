from ultralytics import YOLO
import torch
import time


class OptimizedYOLO:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_data = {}

        if torch.cuda.is_available():
            self.model.to("cuda")
            self.model.fuse()  # Faster inference

    def process(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            classes=[0, 39, 41, 56, 60, 63],  # Person + selected objects
            tracker="bytetrack.yaml",
            verbose=False
        )

        persons = []
        objects = []
        current_time = time.time()

        r = results[0]

        if r.boxes.id is None:
            return persons, objects

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            label = self.model.names[cls]

            if cls == 0:  # PERSON

                obj_id = int(obj_id)
                cy = (y1 + y2) // 2

                if obj_id not in self.person_data:
                    self.person_data[obj_id] = {
                        "start": current_time,
                        "last_y": cy,
                        "dir": "STAY"
                    }

                pdata = self.person_data[obj_id]

                if cy < pdata["last_y"]:
                    pdata["dir"] = "UP"
                elif cy > pdata["last_y"]:
                    pdata["dir"] = "DOWN"

                pdata["last_y"] = cy
                elapsed = int(current_time - pdata["start"])

                persons.append({
                    "id": obj_id,
                    "box": (x1, y1, x2, y2),
                    "time": elapsed,
                    "direction": pdata["dir"]
                })

            else:
                objects.append({
                    "box": (x1, y1, x2, y2),
                    "label": label
                })

        return persons, objects
