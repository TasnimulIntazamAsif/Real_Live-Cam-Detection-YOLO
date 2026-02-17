import cv2

#rtsp_url = "rtsp://admin:boss321%23@192.168.2.42:554/cam/realmonitor?channel=1&subtype=0"
rtsp_url = "rtsp://admin:boss321%23@192.168.2.62:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    cv2.imshow("IP Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
