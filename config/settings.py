# config/settings.py
import os

# General
DEBUG_OUTPUT_DIR = os.getenv('DEBUG_OUTPUT_DIR', 'debug_outputs')
ENABLE_DEBUG_SAVE = os.getenv('ENABLE_DEBUG_SAVE', 'false').lower() == 'true'

# Create directory if enabled
if ENABLE_DEBUG_SAVE:
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)