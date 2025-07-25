import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from deep_sort_realtime.deepsort_tracker import DeepSort
from Distance_esimation import estimate_distance
from Speed_and_Safe_distance import safe_distance_estimation
from query import query_safe_distance_grpc
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture(r"D:\04.Workspace\wrong-way-vehicle-detection-20250725T043341Z-1-001\wrong-way-vehicle-detection\yolov5\input.MP4")
# initialize the video writer object
# writer = create_video_writer(video_cap, "output4.mp4")

####

frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
EGO_POSITION = (frame_width // 2, frame_height)

import os

model_path = os.path.join("best.pt")
print(os.path.exists(model_path))
model = YOLO(model_path)
# load the pre-trained YOLOv8n model
tracker = DeepSort(max_age=50)


while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []
    #####################################
    speed, safe_distance = query_safe_distance_grpc(frame)
    print(f"Safe distance in this frame is : {speed}, {safe_distance}")
    text_speed = f"Speed: {speed}km/h"
    text_safe = f"Safe dist: {safe_distance}m"

    cv2.putText(frame, text_speed, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.putText(frame, text_safe, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    
    # for data in detections.boxes.data.tolist():
    #     # extract the confidence (i.e., probability) associated with the prediction
    #     confidence = data[4]

    #     # filter out weak detections by ensuring the 
    #     # confidence is greater than the minimum confidence
    #     if float(confidence) < CONFIDENCE_THRESHOLD:
    #         continue

    #     # if the confidence is greater than the minimum confidence,
    #     # get the bounding box and the class id
    #     xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
    #     class_id = int(data[5])
    #     # add the bounding box (x, y, w, h), confidence and class id to the results list
    #     results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    # ######################################
    # # TRACKING
    # ######################################

    # # update the tracker with the new detections
    # tracks = tracker.update_tracks(results, frame=frame)
    # # object_centers = []
    # # loop over the tracks
    # for track in tracks:
    #     # if the track is not confirmed, ignore it
    #     if not track.is_confirmed():
    #         continue

    #     # get the track id and the bounding box
    #     track_id = track.track_id
    #     ltrb = track.to_ltrb()

    #     xmin, ymin, xmax, ymax = int(ltrb[0]), int(
    #         ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
    #     # Esimation distance
    #     distance = estimate_distance([xmin, ymin, xmax, ymax], class_id)
    #     # draw the bounding box and the track id
    #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
    #     cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
    #     cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    #     cv2.putText(frame, f"{distance} m", (xmin + 5, ymin - 30),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    # # ####### Line từ xe ego đến các object khác
    #     center_x = (xmin + xmax) // 2
    #     center_y = (ymin + ymax) // 2
    #     object_center = (center_x, center_y)
    #     # object_centers.append(object_center)
    #     cv2.line(frame, EGO_POSITION, object_center, (255, 0, 0), 2)
    #     cv2.circle(frame, object_center, 4, (255, 0, 0), -1)
    # ####
    #     # for i in range(len(object_centers)):
    #     #     for j in range(i + 1, len(object_centers)):
    #     #         pt1 = object_centers[i]
    #     #         pt2 = object_centers[j]
    #     #         cv2.line(frame, pt1, pt2, (0, 165, 255), 1)
    # # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    # writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()


