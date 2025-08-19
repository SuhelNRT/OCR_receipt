# src/debug_saver.py
import os
import json
import shutil
from datetime import datetime
from PIL import Image
from config.settings import DEBUG_OUTPUT_DIR, ENABLE_DEBUG_SAVE

def save_debug_artifacts(
    original_image_path: str,
    cropped_image: Image.Image,
    raw_text: str,
    kv_result: dict,
    prefix: str = ""
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
        if os.path.exists(original_image_path):
            shutil.copy(original_image_path, uploaded_dest)

        # 2. Save cropped image
        cropped_dest = os.path.join(folder_path, "02_cropped.jpg")
        cropped_image.save(cropped_dest, "JPEG", quality=95)

        # 3. Save OCR text
        text_dest = os.path.join(folder_path, "03_ocr_text.txt")
        with open(text_dest, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # 4. Save final JSON output
        json_dest = os.path.join(folder_path, "04_output.json")
        with open(json_dest, "w", encoding="utf-8") as f:
            json.dump(kv_result, f, indent=2, ensure_ascii=False)

        # Optional: Create symlink to latest
        latest_link = os.path.join(DEBUG_OUTPUT_DIR, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        os.symlink(folder_name, latest_link)

        # Log
        from src.utils import log_step
        log_step("debug_save", "completed", folder=folder_name)

    except Exception as e:
        from src.utils import log_step
        log_step("debug_save", "failed", error=str(e))
        # Don't raise — this is optional