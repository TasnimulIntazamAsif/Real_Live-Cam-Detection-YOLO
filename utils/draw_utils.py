import cv2
import time


def draw_boxes(frame, human_boxes, object_boxes):
    for (x1, y1, x2, y2) in human_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for (x1, y1, x2, y2) in object_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame


def draw_info(frame, human_count, object_count, start_time):
    elapsed = int(time.time() - start_time)

    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60

    cv2.putText(frame, f"Human Count: {human_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Object Count: {object_count}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Working Time: {h:02d}:{m:02d}:{s:02d}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return frame
