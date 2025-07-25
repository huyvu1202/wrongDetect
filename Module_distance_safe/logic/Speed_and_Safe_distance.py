import time
from PIL import Image 
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import re
import os
import cv2


muy = 0.8
g = 9.8
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
def get_speed(frame):
    # xmin, ymin, xmax, ymax = 11, 1701, 2549, 1920
    # roi = frame[ymin:ymax, xmin:xmax]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(pil_img)
    speed = extract_speed(text)
    # print(f'Text of frame :{text},  Speed : {speed}')
    return speed

def safe_distance_estimation(frame):
    start = time.perf_counter()

    speed = get_speed(frame)
    safe_distance = 0
    if speed is None or speed ==0:
        print('no speed')
    else:
        speed = float(speed)
    # muy = float(muy)
    # g = float(g)
        safe_distance = speed**2/(2*muy*g)
   
    end = time.perf_counter()

    print(f"🕒 API round-trip time: {(end - start):.3f} seconds")

    return speed, safe_distance
    
    
