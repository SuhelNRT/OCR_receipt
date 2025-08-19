# src/preprocessing.py
import numpy as np
from PIL import Image
from ultralytics import YOLO

def crop_receipt_with_yolo(image: Image.Image, model) -> Image.Image:
    img_np = np.array(image)
    results = model(img_np, imgsz=640, conf=0.25)
    
    masks = results[0].masks
    if masks is None:
        raise ValueError("No receipt detected")

    areas = [np.sum(mask.data.cpu().numpy()) for mask in masks]
    largest_idx = np.argmax(areas)
    x, y, w, h = results[0].boxes.xywh[largest_idx].cpu().numpy().astype(int)
    
    x1, y1 = max(0, x - w//2), max(0, y - h//2)
    x2, y2 = min(image.width, x + w//2), min(image.height, y + h//2)
    
    return image.crop((x1, y1, x2, y2))

def extract_text_with_ocr(image: Image.Image, ocr_reader) -> str:
    img_np = np.array(image)
    results = ocr_reader.readtext(img_np, detail=1)
    
    lines = {}
    for (bbox, text, prob) in results:
        y_pos = bbox[0][1]
        line_key = round(y_pos / 20)
        lines.setdefault(line_key, []).append(text)
    
    return "\n".join([" ".join(line) for k in sorted(lines) for line in lines[k]])