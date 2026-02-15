import cv2
import time
from flask import Flask, render_template, Response
from config import CAMERA_SOURCE, YOLO_MODEL, CONFIDENCE_THRESHOLD
from detector.yolo_detector import YOLODetector
from utils.draw_utils import draw_boxes, draw_info

app = Flask(__name__)

cap = cv2.VideoCapture(CAMERA_SOURCE)
detector = YOLODetector(YOLO_MODEL, CONFIDENCE_THRESHOLD)
start_time = time.time()


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        human_boxes, object_boxes = detector.detect(frame)
        frame = draw_boxes(frame, human_boxes, object_boxes)
        frame = draw_info(frame, len(human_boxes), len(object_boxes), start_time)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    app.run(debug=True)
