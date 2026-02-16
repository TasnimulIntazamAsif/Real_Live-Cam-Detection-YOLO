from ultralytics import YOLO


class YOLOObjectDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_objects(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[
                39,  # bottle
                41,  # cup
                56,  # chair
                60,  # dining table (desk)
                63,  # laptop
            ]
        )

        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]

                objects.append({
                    "box": (x1, y1, x2, y2),
                    "label": label
                })

        return objects