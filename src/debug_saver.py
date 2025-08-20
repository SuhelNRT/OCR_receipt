# src/debug_saver.py
import os
import json
import shutil
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from config.settings import DEBUG_OUTPUT_DIR, ENABLE_DEBUG_SAVE
from src.utils import log_step


def ensure_rgb(image):
    """Convert any image format to RGB for consistent saving"""
    if isinstance(image, np.ndarray):
        # Handle OpenCV BGR format
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        # Handle grayscale
        elif len(image.shape) == 2:
            return np.stack([image]*3, axis=-1)
    
    # Handle PIL images
    if hasattr(image, 'convert'):
        return image.convert('RGB')
    
    raise ValueError(f"Unsupported image format: {type(image)}")

def save_debug_artifacts(
    original_image_path: str,
    cropped_image: Image.Image,
    raw_text: str,
    kv_result: dict,
):
    """
    Save all intermediate artifacts in a timestamped folder.
    Can be called optionally from the pipeline.
    """
    if not ENABLE_DEBUG_SAVE:
        return

    try:
        # Create folder name: timestamp__filename
        filename = os.path.basename(original_image_path)
        safe_name = "".join([c for c in filename if c.isalnum() or c in "._-"])
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        folder_name = f"{timestamp}__{safe_name}"
        folder_path = os.path.join(DEBUG_OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 1. Save uploaded image
        uploaded_dest = os.path.join(folder_path, "01_uploaded.jpg")
        try:
            # Copy original file
            shutil.copy(original_image_path, uploaded_dest)
        except Exception as e:
            log_step("debug_save", "warning", 
                    message="Failed to save uploaded image", error=str(e))

        # 2. Save cropped image
        cropped_dest = os.path.join(folder_path, "02_cropped.jpg")
        try:
            rgb_cropped = ensure_rgb(cropped_image)
            if isinstance(rgb_cropped, np.ndarray):
                cv2.imwrite(cropped_dest, cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2BGR))
            else:  # PIL Image
                rgb_cropped.save(cropped_dest, "JPEG", quality=95)
        except Exception as e:
            log_step("debug_save", "warning", 
                    message="Failed to save cropped image", error=str(e))

        # 3. Save OCR text
        text_dest = os.path.join(folder_path, "03_ocr_text.txt")
        try:
            with open(text_dest, "w", encoding="utf-8") as f:
                f.write(raw_text)
        except Exception as e:
            log_step("debug_save", "warning", 
                    message="Failed to save OCR text", error=str(e))

        # 4. Save final JSON output
        json_dest = os.path.join(folder_path, "04_output.json")
        try:
            with open(json_dest, "w", encoding="utf-8") as f:
                json.dump(kv_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_step("debug_save", "warning", 
                    message="Failed to save output JSON", error=str(e))
            
        # 5. Save metadata
        metadata = {
            "original_filename": filename,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_steps": [
                {"step": "crop", "model": "yolo_receipt_v3"},
                {"step": "ocr", "engine": "easyocr", "languages": ["en"]},
                {"step": "ner", "model": "distilbert_receipt_ner_v2"}
            ],
            "input_size": f"{Image.open(original_image_path).size[0]}x{Image.open(original_image_path).size[1]}",
            "extracted_fields": list(kv_result.keys())
        }
        
        meta_dest = os.path.join(folder_path, "metadata.json")
        try:
            with open(meta_dest, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            log_step("debug_save", "warning", 
                    message="Failed to save metadata", error=str(e))

        # Optional: Create symlink to latest
        latest_link = os.path.join(DEBUG_OUTPUT_DIR, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            os.rename(latest_link, latest_link + ".bak")
        os.symlink(folder_name, latest_link)

        log_step("debug_save", "completed", folder=folder_name)

    except Exception as e:
        log_step("debug_save", "error", 
                error=str(e), error_type=type(e).__name__)