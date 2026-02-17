# 🎥 Real-Time CCTV Detection & Movement Tracking System

A real-time object detection and human movement monitoring system built using YOLOv8,YOLO26N,YOLO26S Flask, and OpenCV.  
This system processes live RTSP CCTV streams, performs object detection with tracking, and displays movement duration along with real-time FPS.

---

## 🚀 Features

- 📡 Live RTSP camera streaming
- 🧠 YOLO-based object detection
- 👤 Human detection with tracking (ByteTrack)
- ⏱ Movement duration tracking (how long a person stays/moves)
- 🔄 Direction detection (UP / DOWN / STAY)
- 📊 Real-time FPS display
- ⚡ Optimized performance (threaded capture + GPU FP16 support)

---

## 🛠 Tech Stack

- Python
- OpenCV
- Flask
- Ultralytics YOLOv8
- PyTorch

---

## 📂 Project Structure

Real_Live-Cam-Detection-YOLO/
│
├── app.py
├── config.py
├── detector/
│   ├── optimized_yolo_fast.py
│   └── threaded_camera.py
├── utils/
│   └── draw_utils.py
├── templates/
│   └── index.html
└── README.md

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

git clone <your-repository-url>  
cd Real_Live-Cam-Detection-YOLO  

### 2️⃣ Create Virtual Environment (Recommended)

Windows:
python -m venv .venv  
.venv\Scripts\activate  

### 3️⃣ Install Dependencies

pip install ultralytics flask opencv-python torch torchvision  

---

## 🎯 Configuration

Open `config.py` and edit:

CAMERA_SOURCE = "rtsp://username:password@ip_address:port/stream"  
YOLO_MODEL = "yolov8n.pt"  
CONFIDENCE_THRESHOLD = 0.5  

Model options:

- yolov8n.pt → Fastest (recommended for real-time)
- yolov8s.pt → Better accuracy
- yolov8m.pt → Higher accuracy, slower
- Custom trained model (e.g., yolo26.pt)

---

## ▶️ Run the Application

python app.py  

Then open in browser:

http://127.0.0.1:5000  

---

## 📊 Performance Optimization

This system includes:

- Threaded RTSP frame capture
- Reduced input resolution (640×360)
- GPU FP16 acceleration (if CUDA available)
- ByteTrack object tracking
- Stable averaged FPS display
- Optimized JPEG streaming quality

Expected Performance:

CPU Only: 15–25 FPS  
Mid-range GPU: 40–60 FPS  
High-end GPU: 60+ FPS  

---

## 🧠 Custom Model Training (Optional)

To train your own YOLO model (example: yolo26.pt):

yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 name=yolo26  

After training, you will find:

runs/detect/yolo26/weights/best.pt  

Rename best.pt to:

yolo26.pt  

Place it in your project folder and update config.py.

---

## 📌 Use Cases

- Smart office monitoring
- Security surveillance
- Human activity analytics
- Loitering detection
- Workplace safety monitoring

---

## 📜 License

This project is for educational and research purposes only.

---

## 👨‍💻 Author

Developed as a real-time AI surveillance system using YOLO and Flask.
