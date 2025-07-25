import cv2
import grpc
import time
import distance_pb2
import distance_pb2_grpc


def query_safe_distance_grpc(frame):
    # Resize frame if needed
    xmin, ymin, xmax, ymax = 11, 1701, 2549, 1920
    roi = frame[ymin:ymax, xmin:xmax]
    # Encode to WebP or JPEG
    t0 = time.perf_counter()
    success, img_encoded = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        print("[Encoding Error] Failed to encode frame")
        return 0, 0
    t1 = time.perf_counter()

    # Setup gRPC channel
    try:
        
        if img_encoded is None or len(img_encoded) == 0:
            print("[Warning] Encoded image is empty")
            return 0, 0

# Global setup (outside function)
        channel = grpc.insecure_channel('192.168.1.8:50051')
        stub = distance_pb2_grpc.DistanceServiceStub(channel)

        # Send image
        request = distance_pb2.FrameRequest(image=img_encoded.tobytes())
        response = stub.Estimate(request, timeout=0.5)       
        t2 = time.perf_counter()

        print(f"⏱️ Encoding: {t1 - t0:.3f}s | gRPC transfer: {t2 - t1:.3f}s | Total: {t2 - t0:.3f}s")
        return response.speed, response.safe_distance

    except Exception as e:
        t2 = time.perf_counter()
        print(f"[Exception] gRPC call failed: {e}")
        print(f"🕒 Attempt duration: {t2 - t0:.3f}s")
        return 0, 0
