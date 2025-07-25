import cv2
import grpc
import time
from concurrent.futures import ThreadPoolExecutor

from gRPC import distance_pb2, distance_pb2_grpc
from gRPC import detection_pb2, detection_pb2_grpc

def query_multi_module(frame):
    xmin, ymin, xmax, ymax = 11, 1701, 2549, 1920
    roi = frame[ymin:ymax, xmin:xmax]
    t0 = time.perf_counter()
    success, img_encoded1 = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 80])
    success, img_encoded2 = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not success or img_encoded1 is None or len(img_encoded1) == 0:
        print("[Encoding Error] Failed to encode frame")
        return None, None
    t1 = time.perf_counter()

    def query_distance():
        frame_bytes = img_encoded1.tobytes()
        channel = grpc.insecure_channel('192.168.1.8:50051')
        stub = distance_pb2_grpc.DistanceServiceStub(channel)
        request = distance_pb2.FrameRequest(image=frame_bytes)
        return stub.Estimate(request, timeout=1.0)

    def query_detection():
        frame_bytes = img_encoded2.tobytes()
        channel = grpc.insecure_channel('192.168.1.8:50052')
        stub = detection_pb2_grpc.DetectionServiceStub(channel)
        request = detection_pb2.FrameRequest(image=frame_bytes)
        return stub.Detect(request, timeout=1.0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dist = executor.submit(query_distance)
        future_detect = executor.submit(query_detection)

        try:
            response_dist = future_dist.result(timeout=1.0)
            response_detect = future_detect.result(timeout=1.0)
            t2 = time.perf_counter()

            print(f"⏱️ Encoding: {t1 - t0:.3f}s | Total round-trip: {t2 - t0:.3f}s")
            return response_dist, response_detect

        except Exception as e:
            print(f"[Parallel gRPC Error] {e}")
            return None, None
