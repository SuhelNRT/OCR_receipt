# src/validation.py
import os

def validate_image_file(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("Unsupported image format")