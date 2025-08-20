# config/settings.py
import os

# General
DEBUG_OUTPUT_DIR = os.getenv('DEBUG_OUTPUT_DIR', 'debug_outputs')
ENABLE_DEBUG_SAVE = os.getenv('ENABLE_DEBUG_SAVE', 'false').lower() == 'true'

# Create directory if enabled
if ENABLE_DEBUG_SAVE:
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# Model paths
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/best_inst_seg.onnx')  #best_inst_seg
NER_MODEL_PATH = os.getenv('NER_MODEL_PATH', 'models/distilbert_receipts_best')
OCR_LANGUAGES = os.getenv('OCR_LANGUAGES', 'en').split(',')

# ONNX-specific settings
YOLO_ONNX_INPUT_SIZE = 640  # Must match export size  # (640, 640)
YOLO_ONNX_CONF_THRESHOLD = float(os.getenv('YOLO_ONNX_CONF_THRESHOLD', '0.25'))

# # Create directories
# os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True) if os.path.dirname(YOLO_MODEL_PATH) else None
# os.makedirs(os.path.dirname(NER_MODEL_PATH), exist_ok=True) if os.path.dirname(NER_MODEL_PATH) else None