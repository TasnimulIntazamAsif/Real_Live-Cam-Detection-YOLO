import cv2
import time
from collections import deque
from flask import Flask, render_template, Response
from config import CAMERA_SOURCE, YOLO_MODEL, CONFIDENCE_THRESHOLD
from detector.optimized_yolo_fast import OptimizedYOLOFast
from detector.threaded_camera import VideoStream
from utils.draw_utils import draw_persons, draw_objects
from threading import Thread, Lock

app = Flask(__name__)

# ------------------ Camera ------------------
vs = VideoStream(CAMERA_SOURCE, width=480, height=270)  # smaller = faster

# ------------------ YOLO ------------------
detector = OptimizedYOLOFast(YOLO_MODEL, CONFIDENCE_THRESHOLD)

# Shared frame and results
latest_frame = None
latest_persons = []
latest_objects = []
frame_lock = Lock()

# FPS calculation
fps_buffer = deque(maxlen=30)
prev_time = 0

# ------------------ YOLO THREAD ------------------
def yolo_worker():
    global latest_frame, latest_persons, latest_objects
    while True:
        if latest_frame is None:
            continue
        # Take the latest frame, skip if YOLO is busy
        frame_copy = None
        with frame_lock:
            frame_copy = latest_frame
        if frame_copy is None:
            continue
        persons, objects = detector.process(frame_copy)
        with frame_lock:
            latest_persons = persons
            latest_objects = objects

Thread(target=yolo_worker, daemon=True).start()

# ------------------ STREAM GENERATOR ------------------
def generate_frames():
    global prev_time, latest_frame
    while True:
        success, frame = vs.read()
        if not success or frame is None:
            continue

        # Store latest frame for YOLO thread
        with frame_lock:
            latest_frame = frame

        # Draw YOLO results
        with frame_lock:
            persons_copy = latest_persons
            objects_copy = latest_objects
        frame_drawn = frame.copy()
        frame_drawn = draw_persons(frame_drawn, persons_copy)
        frame_drawn = draw_objects(frame_drawn, objects_copy)

        # FPS calculation
        current_time = time.time()
        if prev_time != 0:
            fps = 1 / (current_time - prev_time)
            fps_buffer.append(fps)
        prev_time = current_time
        avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0

        cv2.putText(frame_drawn, f"FPS: {int(avg_fps)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode frame faster
        ret, buffer = cv2.imencode(".jpg", frame_drawn, [cv2.IMWRITE_JPEG_QUALITY, 40])
        frame_bytes = buffer.tobytes()

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"

# ------------------ FLASK ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False, threaded=True)
