import argparse
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import cv2
import math
import numpy as np
from collections import defaultdict
from collections import deque

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_img_size, check_imshow, check_requirements,
                        increment_path, print_args, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from pyimagesearch.centroidtracker import CentroidTracker


ANGLE_CHANGE_THRESHOLD = 5.0  
MAX_ANGLE_HISTORY = 10        
CROSSING_DISTANCE_THRESH = 50 

def normalize_angle(angle):
    """Convert angle to 0-360 range"""
    return angle % 360

def detect_crossing(angle_history):
    """Improved crossing detection with angle wrapping handling"""
    if len(angle_history) < 3:
        return False
        
    total_change = 0
    for i in range(1, len(angle_history)):
        delta = angle_history[i-1] - angle_history[i]
        if delta > 180: delta -= 360
        elif delta < -180: delta += 360
        total_change += delta
    
    avg_change = total_change / (len(angle_history)-1)
    return abs(avg_change) > ANGLE_CHANGE_THRESHOLD

@torch.no_grad()
def run(
        weights='best.pt',
        source='data/images',
        data='data.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project='runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
):
    # Initialize
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = YOLO(weights[0])
    names = model.model.names
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        bs = 1

    # Trackers and data structures
    ct = CentroidTracker()  # Centroid tracker
    angle_history = defaultdict(lambda: deque(maxlen=MAX_ANGLE_HISTORY))
    vid_path, vid_writer = [None] * bs, [None] * bs
    prev_centroids = {}  # Store previous centroids for direction calculation

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        # Inference
        results = model.predict(source=im, conf=conf_thres, iou=iou_thres, device=device)
        pred = results[0].boxes.data

        # Process detections
        p, im0 = path, im0s.copy()
        p = Path(p)
        save_path = str(save_dir / p.name)
        txt_path = str(save_dir / 'labels' / p.stem)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        imc = im0.copy() if save_crop else im0
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        # Frame dimensions and positions
        FRAME_WIDTH = im0.shape[1]
        FRAME_HEIGHT = im0.shape[0]
        EGO_POSITION = (FRAME_WIDTH // 2, FRAME_HEIGHT)  # Bottom center
        
        # ROI Definitions
        ROI_X_MIN = int(FRAME_WIDTH * 0.50)
        ROI_X_MAX = int(FRAME_WIDTH * 0.95)
        ROI_Y_MIN = int(FRAME_HEIGHT * 0.10)
        ROI_Y_MAX = int(FRAME_HEIGHT * 0.70)

        # Prepare object lists
        all_objects = []  # List to store all detected objects
        rects = []  # For centroid tracker

        if len(pred):
            pred = pred.clone()
            pred_coords = scale_coords(im.shape[2:], pred[:, :4].clone(), im0.shape).round()
            pred[:, :4] = pred_coords

            for *xyxy, conf, cls in reversed(pred):
                x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                
                # Add to all objects list
                all_objects.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (box_center_x, box_center_y),
                    'class': int(cls.item()),
                    'conf': conf.item()
                })
                
                # Add rectangle for centroid tracking
                rects.append((x1, y1, x2, y2))

                # Distance estimation (uncomment if needed)
                # distance = estimate_distance([x1, y1, x2, y2], int(cls.item()))
                # cv2.putText(im0, f"{distance}m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if save_txt:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls.item(), *xywh, conf.item()) if save_conf else (cls.item(), *xywh)
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or save_crop or view_img:
                    c = int(cls.item())
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[int(cls.item())] / f'{p.stem}.jpg', BGR=True)

        im0 = annotator.result()

        # Update centroid tracker
        objects, CY1, CY2 = ct.update(rects)

        # Store current centroids for next frame comparison
        current_centroids = {}
        crossing_status = {}
        direction_status = {}

        # Process all objects
        for (objectID, centroid) in objects.items():
            current_centroids[objectID] = centroid
            
            # Calculate angle between object and ego position
            dx = centroid[0] - EGO_POSITION[0]
            dy = EGO_POSITION[1] - centroid[1]  # Inverted Y axis
            angle = normalize_angle(math.degrees(math.atan2(dy, dx)))
            
            # Update angle history
            angle_history[objectID].append(angle)
            
            # Detect crossing based on angle change
            crossing = False
            if len(angle_history[objectID]) >= 3:
                crossing = detect_crossing(angle_history[objectID])
            crossing_status[objectID] = crossing

            # Visualization
            color = (0, 0, 255) if crossing else (255, 0, 0)
            cv2.line(im0, EGO_POSITION, centroid, color, 1)
            cv2.circle(im0, centroid, 4, color, -1)
            
            # Display angle and crossing status
            # cv2.putText(im0, f"{angle:.1f}°", (centroid[0] + 10, centroid[1] + 10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if crossing:
                cv2.putText(im0, "CROSSING!", (centroid[0], centroid[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Determine direction for objects in ROI
        for objectID, centroid in objects.items():
            direction = "Unknown"
            color = (255, 255, 255)  # Default white
            
            # Check if object is in ROI
            in_roi = (ROI_X_MIN <= centroid[0] <= ROI_X_MAX and 
                      ROI_Y_MIN <= centroid[1] <= ROI_Y_MAX)
            
            if in_roi and objectID in prev_centroids:
                prev_centroid = prev_centroids[objectID]
                cy1 = prev_centroid[1]
                cy2 = centroid[1]
                
                # New direction logic
                if cy2 <= cy1:
                    if crossing_status.get(objectID, False):
                        direction = "Right (Crossing)"
                    else:
                        direction = "Right"
                    color = (0, 255, 0)  # Green
                else:
                    direction = "WRONG WAY"
                    color = (0, 0, 255)  # Red
                
                # Display direction
                cv2.putText(im0, direction, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            direction_status[objectID] = direction
            cv2.circle(im0, (centroid[0], centroid[1]), 3, color, -1)

        # Update previous centroids for next frame
        prev_centroids = current_centroids

        # Draw semi-transparent ROI
        roi_mask = im0.copy()
        cv2.rectangle(roi_mask, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX), (0, 180, 0), -1)
        im0 = cv2.addWeighted(im0, 0.7, roi_mask, 0.3, 0)
        
        # Draw ROI border
        cv2.rectangle(im0, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX), (0, 255, 0), 1)

        # Stream/view results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)

        # Save results
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path[0] != save_path:
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0)

# ... (rest of the code remains unchanged)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
