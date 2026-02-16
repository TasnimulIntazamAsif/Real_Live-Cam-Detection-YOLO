import cv2
import time
import math
from flask import Flask, render_template, Response
from config import CAMERA_SOURCE, YOLO_MODEL, CONFIDENCE_THRESHOLD
from detector.yolo_detector import YOLODetector
from utils.draw_utils import draw_boxes, draw_info

app = Flask(__name__)

# Use FFMPEG for RTSP stability
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)

detector = YOLODetector(YOLO_MODEL, CONFIDENCE_THRESHOLD)
start_time = time.time()

# Store previous object centers
previous_centers = {}
MOVEMENT_THRESHOLD = 20  # pixels


def calculate_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_frames():
    global previous_centers

    while True:
        success, frame = cap.read()
        if not success:
            break

        human_boxes, object_boxes = detector.detect(frame)

        current_centers = {}
        movement_data = []

        # Combine both human + objects
        all_boxes = human_boxes + object_boxes

        for idx, box in enumerate(all_boxes):

            center = calculate_center(box)
            current_centers[idx] = center

            if idx in previous_centers:
                distance = calculate_distance(center, previous_centers[idx])
                moving = distance > MOVEMENT_THRESHOLD
            else:
                moving = False

            movement_data.append((box, moving))

        previous_centers = current_centers

        frame = draw_boxes(frame, movement_data)
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
