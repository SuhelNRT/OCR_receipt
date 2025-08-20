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
    