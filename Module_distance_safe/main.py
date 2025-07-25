import grpc
from concurrent import futures
import distance_pb2
import distance_pb2_grpc
import numpy as np
import cv2
from logic.Speed_and_Safe_distance import safe_distance_estimation
class DistanceService(distance_pb2_grpc.DistanceServiceServicer):
    def Estimate(self, request, context):
        try:
            img_bytes = request.image
            if not img_bytes:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Received empty image payload")
                return distance_pb2.EstimateResponse(speed=0.0, safe_distance=0.0)

            npimg = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None or img.size == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Failed to decode image (empty result)")
                return distance_pb2.EstimateResponse(speed=0.0, safe_distance=0.0)

            speed, safe_distance = safe_distance_estimation(img)
            return distance_pb2.EstimateResponse(speed=speed, safe_distance=safe_distance)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return distance_pb2.EstimateResponse(speed=0.0, safe_distance=0.0)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    distance_pb2_grpc.add_DistanceServiceServicer_to_server(DistanceService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("🚀 gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
