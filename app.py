import cv2
from flask import Flask, render_template, Response
from config import CAMERA_SOURCE, YOLO_MODEL, CONFIDENCE_THRESHOLD
from detector.yolo_tracker import YOLOTracker
from detector.yolo_objects import YOLOObjectDetector
from utils.draw_utils import draw_persons, draw_objects

app = Flask(__name__)

cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)

person_tracker = YOLOTracker(YOLO_MODEL, CONFIDENCE_THRESHOLD)
object_detector = YOLOObjectDetector(YOLO_MODEL, CONFIDENCE_THRESHOLD)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        persons = person_tracker.track_persons(frame)
        objects = object_detector.detect_objects(frame)

        frame = draw_persons(frame, persons)
        frame = draw_objects(frame, objects)

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


if __name__ == "__main__":
    app.run(debug=True)