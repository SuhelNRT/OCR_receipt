# src/utils.py
import logging
import os
from PIL import Image
from datetime import datetime
from contextlib import contextmanager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/receipt_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("receipt_processor")

def log_step(step_name: str, status: str, **kwargs):
    """
    Unified logging for pipeline steps
    """
    log_entry = {
        "step": step_name,
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs
    }
    msg = " | ".join([f"{k}={v}" for k, v in log_entry.items()])
    logger.info(msg)


@contextmanager
def step_logger(step_name: str, **context):
    """
    Context manager to log step entry, exit, and errors
    """
    start_time = datetime.utcnow()
    log_step(step_name, "started", **context)
    
    try:
        yield
        duration = (datetime.utcnow() - start_time).total_seconds()
        log_step(step_name, "completed", duration_sec=round(duration, 2), **context)
    
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        log_step(
            step_name,
            "failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_sec=round(duration, 2),
            **context
        )
        raise 

def load_image_from_path(image_path: str) -> Image.Image:
    try:
        return Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

def cleanup_temp_file(filepath: str):
    if os.path.exists(filepath):
        os.remove(filepath)