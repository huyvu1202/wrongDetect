from PIL import Image
import pytesseract
import cv2
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_speed(text):
    """
    Hàm trích tốc độ từ chuỗi text OCR, có xử lý lỗi nhận dạng như 'Okm/h' → '0km/h'
    Trả về: int (số km/h) hoặc None nếu không tìm thấy
    """
    if not text:
        return None

    text = re.sub(r'\bO(?=km/h)', '0', text)  # 'Okm/h' → '0km/h'
    text = re.sub(r'\bQ(?=km/h)', '0', text)  # 'Qkm/h' → '0km/h'
    text = re.sub(r'\bB(?=km/h)', '8', text)  # 'Bkm/h' → '8km/h'

    text = re.sub(r'(\d)\s+km/h', r'\1km/h', text)
    match = re.search(r'\b(\d{1,3})km/h\b', text)
    if match:
        return int(match.group(1))
    else:
        return None

def orc_speed():
    path = r"D:\04.Workspace\wrong-way-vehicle-detection-20250725T043341Z-1-001\wrong-way-vehicle-detection\yolov5\input3.MP4"
    cap = cv2.VideoCapture(path)

    xmin, ymin, xmax, ymax = 11, 1701, 2549, 1920
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 100 == 0:
            roi = frame[ymin:ymax, xmin:xmax]
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img)
            print(text)
            speed = extract_speed(text)
            print(f'Frame {frame_idx} : {speed} ')
            print('---------------------------------------------------')

        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()

orc_speed()