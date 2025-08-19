# src/model_manager.py
from ultralytics import YOLO
import easyocr
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class ModelManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        self.yolo = YOLO(r"models\best_inst_seg.pt")  # can change for onnx
        self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())   # change with trained model
        self.ner_tokenizer = AutoTokenizer.from_pretrained("models/distilbert_receipts_best")
        self.ner_model = AutoModelForTokenClassification.from_pretrained("models/distilbert_receipts_best")

model_manager = ModelManager()