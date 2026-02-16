import cv2
import time


def draw_boxes(frame, movement_data):

    for box, moving in movement_data:
        x1, y1, x2, y2 = box

        if moving:
            color = (0, 0, 255)  # Red for moving
            label = "Moving"
        else:
            color = (0, 255, 0)  # Green for static
            label = "Static"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    return frame


def draw_info(frame, human_count, object_count, start_time):
    elapsed_time = int(time.time() - start_time)

    cv2.putText(frame, f"Humans: {human_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.putText(frame, f"Objects: {object_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.putText(frame, f"Time: {elapsed_time}s",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    return frame
