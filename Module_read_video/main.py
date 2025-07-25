import datetime
import cv2
import os
from helper import create_video_writer
from logic.query import query_multi_module

# Constants
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Initialize video capture and writer
video_cap = cv2.VideoCapture(r"data\input.MP4")
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = create_video_writer(video_cap, "output_with_objects.mp4")
EGO_POSITION = (frame_width // 2, frame_height)

while True:
    start = datetime.datetime.now()
    ret, frame = video_cap.read()
    if not ret:
        break

    results = []
    dist_resp, detect_resp = query_multi_module(frame)

    if dist_resp and detect_resp:
        # 🟡 Display driving-related info
        speed = dist_resp.speed
        safe_distance = dist_resp.safe_distance
        text_speed = f"Speed: {speed}km/h"
        text_safe = f"Safe dist: {safe_distance}m"

        cv2.putText(frame, text_speed, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.putText(frame, text_safe, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # 🟢 Draw detected objects
        for obj in detect_resp.objects:
            if obj.confidence < CONFIDENCE_THRESHOLD:
                continue

            x = obj.bbox.xmin
            y = obj.bbox.ymin
            w = obj.bbox.width
            h = obj.bbox.height
            x2 = x + w
            y2 = y + h

            class_id = obj.class_id
            track_id = obj.track_id
            confidence = obj.confidence

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), GREEN, 2)
            # Add label
            label = f"ID:{track_id} | C:{class_id} | {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    else:
        print("❌ One or both modules failed")

    end = datetime.datetime.now()
    # ⏱ Frame timing + FPS overlay
    processing_time_ms = (end - start).total_seconds() * 1000
    fps = 1 / (end - start).total_seconds()
    print(f"Time to process 1 frame: {processing_time_ms:.0f} ms")
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # 🔁 Show frame & save to output video
    cv2.imshow("Frame", frame)
    writer.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

# ✅ Release resources
video_cap.release()
writer.release()
cv2.destroyAllWindows()
