from ultralytics import YOLO
import torch


class YOLODetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.confidence = confidence

        if torch.cuda.is_available():
            self.model.to("cuda")

    def detect(self, frame):
        results = self.model(frame, stream=True)

        human_boxes = []
        object_boxes = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < self.confidence:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 0:      # Human
                    human_boxes.append((x1, y1, x2, y2))
                else:             # Other objects
                    object_boxes.append((x1, y1, x2, y2))

        return human_boxes, object_boxes
