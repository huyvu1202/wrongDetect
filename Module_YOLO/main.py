import grpc
from concurrent import futures
from gRPC import detection_pb2
from gRPC import detection_pb2_grpc

import cv2
import numpy as np
from ultralytics import YOLO

class DetectionService(detection_pb2_grpc.DetectionServiceServicer):
    def __init__(self):
        self.model = YOLO('best.pt')

    def Detect(self, request, context):
        image_bytes = request.image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = self.model(frame)
        response = detection_pb2.ObjectList()

        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            xmin = int(xyxy[0])
            ymin = int(xyxy[1])
            width = int(xyxy[2] - xyxy[0])
            height = int(xyxy[3] - xyxy[1])
            center_x = xmin + width // 2
            center_y = ymin + height // 2

            obj = detection_pb2.ObjectInfo(
                track_id=0,
                class_id=cls,
                confidence=conf,
                bbox=detection_pb2.BBox(xmin=xmin, ymin=ymin, width=width, height=height),
                distance=0.0,
                center_x=center_x,
                center_y=center_y
            )
            response.objects.append(obj)

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    detection_pb2_grpc.add_DetectionServiceServicer_to_server(DetectionService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("Detection gRPC server is running at port 50052")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
