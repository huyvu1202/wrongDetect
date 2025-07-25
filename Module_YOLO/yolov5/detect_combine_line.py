import argparse
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import cv2

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_img_size, check_imshow, check_requirements,
                        increment_path, print_args, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from pyimagesearch.centroidtracker import CentroidTracker

from Distance_esimation import estimate_distance
from Speed_and_Safe_distance import safe_distance_estimation
import math
angle_history_template = {}
ct = CentroidTracker()

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
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = YOLO(weights[0])
    names = model.model.names
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        bs = 1

    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        results = model.predict(source=im, conf=conf_thres, iou=iou_thres, device=device)
        pred = results[0].boxes.data

        rects = []
        # rects_left = []
        p, im0 = path, im0s.copy()
        p = Path(p)
        save_path = str(save_dir / p.name)
        txt_path = str(save_dir / 'labels' / p.stem)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        imc = im0.copy() if save_crop else im0
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        FRAME_WIDTH = im0.shape[1]
        FRAME_HEIGHT = im0.shape[0]
        ROI_X_MIN = int(FRAME_WIDTH * 0.50)
        ROI_X_MAX = int(FRAME_WIDTH * 0.95)
        ROI_Y_MIN = int(FRAME_HEIGHT * 0.10)
        ROI_Y_MAX = int(FRAME_HEIGHT * 0.70)
        EGO_POSITION = (FRAME_WIDTH // 2, FRAME_HEIGHT)
        
        # ROI_X_MIN_left = int(FRAME_WIDTH * 0.1)
        # ROI_X_MAX_left = int(FRAME_WIDTH * 0.48)
        # ROI_Y_MIN_left = int(FRAME_HEIGHT * 0.10)
        # ROI_Y_MAX_left = int(FRAME_HEIGHT * 0.70)

        # Estimate speed and safe distance
        # speed, safe_distance = safe_distance_estimation(im0)
        # text_speed = f"Speed: {speed}km/h"
        # text_safe = f"Safe dist: {safe_distance}m"
        # cv2.putText(im0, text_speed, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        # cv2.putText(im0, text_safe, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        if len(pred):
            pred = pred.clone()
            pred_coords = scale_coords(im.shape[2:], pred[:, :4].clone(), im0.shape).round()
            pred[:, :4] = pred_coords

            for *xyxy, conf, cls in reversed(pred):
                x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                cv2.putText(im0, f"({int(box_center_x)}, {int(box_center_y)})",
                (int(box_center_x) + 10, int(box_center_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4)

                # Estimate distance
                distance = estimate_distance([x1, y1, x2, y2], int(cls.item()))
                cv2.putText(im0, f"{distance}m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw line from ego to object
                object_center = (int(box_center_x), int(box_center_y))
                cv2.line(im0, EGO_POSITION, object_center, (255, 0, 0), 1)
                cv2.circle(im0, object_center, 4, (255, 0, 0), -1)
                # Xét vùng ROI bên phải
                if ROI_X_MIN <= box_center_x <= ROI_X_MAX and ROI_Y_MIN <= box_center_y <= ROI_Y_MAX:
                    rects.append((x1, y1, x2, y2))
                # Xét vùng ROI bên trái
                # if ROI_X_MIN_left <= box_center_x <= ROI_X_MAX_left and ROI_Y_MIN_left <= box_center_y <= ROI_Y_MAX_left:
                #     rects_left.append((x1, y1, x2, y2))
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

        # objects, CY1, CY2 = ct.update(rects)
        # for objectID, centroid in objects.items():
        #     cy1, cy2 = CY1[objectID], CY2[objectID]
        #     direction = 'right' if cy2 <= cy1 else 'wrong'
        #     color = (0, 255, 0) if direction == 'right' else (0, 0, 255)
        #     cv2.putText(im0, direction, (centroid[0] - 10, centroid[1] - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #     cv2.circle(im0, (centroid[0], centroid[1]), 3, color, -1)
        objects, CY1, CY2 = ct.update(rects)

        for objectID, centroid in objects.items():
            cy1, cy2 = CY1[objectID], CY2[objectID]
            direction = 'right' if cy2 <= cy1 else 'wrong'
            color = (0, 255, 0) if direction == 'right' else (0, 0, 255)
            cv2.putText(im0, direction, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(im0, (centroid[0], centroid[1]), 3, color, -1)

            # === TÍNH GÓC GIỮA XE EGO VÀ OBJECT ===
            dx = centroid[0] - EGO_POSITION[0]
            dy = EGO_POSITION[1] - centroid[1]  # y tính từ top xuống
            angle = math.degrees(math.atan2(dy, dx))# radians to degrees
            print(f'----------------Angle : {angle}     -------------------')

            if objectID not in angle_history_template:
                angle_history_template[objectID] = []
            angle_history_template[objectID].append(angle)

            # 10 frame gần nhất
            if len(angle_history_template[objectID]) > 10:
                angle_history_template[objectID] = angle_history_template[objectID][-10:]

            # PHÁT HIỆN OBJECT BĂNG QUA ĐƯỜNG DỰA VÀO GÓC GIẢM DẦN
            angles = angle_history_template[objectID]
            crossing = False
            if len(angles) >= 3 and all(earlier > later for earlier, later in zip(angles, angles[1:])):
                crossing = True

            if crossing:
                cv2.putText(im0, "Crossing!", (centroid[0] + 10, centroid[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.line(im0, EGO_POSITION, (centroid[0], centroid[1]), (0, 0, 255), 2)
            else:
                cv2.line(im0, EGO_POSITION, (centroid[0], centroid[1]), (255, 0, 0), 1)
        cv2.rectangle(im0, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX), (0, 255, 0), 2)
        # cv2.rectangle(im0, (ROI_X_MIN_left, ROI_Y_MIN_left), (ROI_X_MAX_left, ROI_Y_MAX_left), (0, 255, 0), 2)
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)

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
