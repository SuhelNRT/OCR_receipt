# src/preprocessing.py
import numpy as np
from PIL import Image
import cv2

def preprocess_image(image, input_size):
    """Convert PIL image to ONNX input tensor"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img_np, (new_w, new_h))
    
    pad_h = input_size[0] - new_h
    pad_w = input_size[1] - new_w
    padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    img_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    return img_tensor, (w, h), (new_w, new_w), (pad_w, pad_h)

def preprocess_image_new(image, img_size):
    """Resize directly to IMG_SIZE without padding"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB
        image = image[:, :, ::-1]

    resized = cv2.resize(image, (img_size, img_size))
    blob = resized.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW  #resized
    # blob = blob.astype(np.float32) / 255.0 
    blob = np.expand_dims(blob, axis=0)  # Add batch dim
    return blob

def xywh2xyxy(boxes, orig_w, orig_h, img_size):
    """Convert center-x, center-y, w, h to x1, y1, x2, y2 at original scale"""
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    # Unpack coordinates
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = (cx- w / 2) * orig_w / img_size
    y1 = (cy - h / 2) * orig_h / img_size
    x2 = (cx + w / 2) * orig_w / img_size
    y2 = (cy + h / 2) * orig_h / img_size

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(int)
    return boxes_xyxy


def postprocess_output_new(det, yolo_conf):
    if det.ndim == 1:
        det = det.reshape(1, -1)

    # Check if we have detections
    if det.shape[0] == 0:
        return None

    boxes = det[:, :4]  # cx, cy, w, h
    obj_conf = det[:, 4:5]
    class_conf = det[:, 5:]
    scores = (obj_conf * class_conf.max(axis=1, keepdims=True)).flatten()
    
    # Find best detection above threshold
    valid_indices = np.where(scores >= yolo_conf)[0]
    if len(valid_indices) == 0:
        return None
    
    best_idx = valid_indices[np.argmax(scores[valid_indices])]
    best_box = boxes[best_idx:best_idx+1]

    return best_box
    
    
    



def postprocess_output(output, orig_size, resized_size, padding, conf_threshold):
    """Process ONNX output to get bounding boxes"""
    # Simplified for detection-only output
    # Adjust based on your actual ONNX output format
    try:
        dets = output[0] if output.ndim == 3 else output
        
        boxes = []
        scores = []
        class_ids = []
        
        for row in dets:
            if row[4] >= conf_threshold:
                cx, cy, w, h = row[:4]
                score = row[4]
                class_id = int(np.argmax(row[5:]))
                
                # Convert to absolute coords
                x1 = (cx - w/2) * orig_size[0] / resized_size[0]
                y1 = (cy - h/2) * orig_size[1] / resized_size[1]
                x2 = (cx + w/2) * orig_size[0] / resized_size[0]
                y2 = (cy + h/2) * orig_size[1] / resized_size[1]
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)
        
        return boxes, [], scores, class_ids  # masks not used here
        
    except Exception as e:
        raise ValueError(f"Postprocessing failed: {str(e)}")
    

def find_best_resize_factor_adaptive(
    image, 
    reader, 
    min_confidence: float = 0.1,
    min_detections: int = 2,
    step: float = 0.2,
    max_steps: int = 2,
    improvement_threshold: float = 0.01
):
    """
    Adaptive resize factor selection with reduced OCR calls.
    Returns:
        best_text   -> full OCR text
        best_factor -> resize factor chosen
        best_conf   -> average confidence
        best_resized-> best resized image
    """
    # image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    h, w = image.shape[:2]
    results = {}
    best_text = ''
    best_factor = 1
    best_ocr = ''
    best_conf = 0.1

    def ocr_and_evaluate(img):
        try:
            ocr_result = reader.readtext(img, detail=1, paragraph=False)
            valid = [det for det in ocr_result if det[2] >= min_confidence]
            avg_conf = np.mean([det[2] for det in valid]) if len(valid) >= min_detections else 0.0
            return avg_conf, len(valid), ocr_result
        except Exception:
            return 0.0, 0, []

    def resize_and_eval(factor):
        if factor in results:   # skip if already computed
            return results[factor]
        new_size = (int(w * factor), int(h * factor))
        interp = cv2.INTER_LANCZOS4 if factor > 1 else cv2.INTER_AREA
        resized = cv2.resize(image, new_size, interpolation=interp)
        avg_conf, num_dets, ocr_result = ocr_and_evaluate(resized)
        results[factor] = (avg_conf, num_dets, resized, ocr_result)
        return results[factor]

    # --- Phase 1: Start with original size ---
    best_factor = 1.0
    best_conf, _, best_resized, best_ocr = resize_and_eval(best_factor)

    # --- Phase 2: Explore up and down adaptively ---
    for direction in [-1, 1]:  # smaller, larger
        factor = 1.0 + direction * step
        steps_done = 0
        while steps_done < max_steps:
            conf, _, resized, ocr_result = resize_and_eval(factor)
            if conf > best_conf + improvement_threshold:
                best_conf, best_factor, best_resized, best_ocr = conf, factor, resized, ocr_result
                factor += direction * step / 2   # keep exploring in same direction
            else:
                break  # stop exploring if no improvement
            steps_done += 1

    # --- Final best text ---
    best_text = "\n".join([det[1] for det in best_ocr])

    return best_text, best_factor #best_text, best_factor, best_conf, best_resized 