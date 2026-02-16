import cv2


def format_time(sec):
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_persons(frame, persons):
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        label = f"ID:{p['id']} | {format_time(p['time'])} | {p['direction']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def draw_objects(frame, objects):
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        label = o["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame