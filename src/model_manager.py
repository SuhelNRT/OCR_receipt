import onnxruntime as ort
import easyocr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import os
import numpy as np
import time
import logging
import re
import time


# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/receipt_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_manager")

# import Settings
from config.settings import (
    YOLO_MODEL_PATH,
    NER_MODEL_PATH,
    OCR_LANGUAGES,
    YOLO_ONNX_INPUT_SIZE,
    YOLO_ONNX_CONF_THRESHOLD,
    CUSTOM_EASYOCR_MODEL_DIR,
    CUSTOM_EASYOCR_NETWORK_NAME
)

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # cls._instance._load_models()
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._yolo = None
        self._ocr = None
        self._ner_tokenizer = None
        self._ner_model = None
        self._ner_pipeline = None
        self._initialized = True

    # def _load_models(self):
    #     # self.yolo = YOLO(r"models\best_inst_seg.pt")  # can change for onnx
    #     self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())   # change with trained model
    #     self.ner_tokenizer = AutoTokenizer.from_pretrained("models/distilbert_receipts_best")
    #     self.ner_model = AutoModelForTokenClassification.from_pretrained("models/distilbert_receipts_best")

    @property
    def yolo(self):
        if self._yolo is None:
            logger.info("Initializing ONNX YOLO model for receipt detection...")
            start_time = time.time()

            try:
                # Check if model file exists
                if not os.path.exists(YOLO_MODEL_PATH):
                    raise FileNotFoundError(f"YOLO ONNX model not found: {YOLO_MODEL_PATH}")

                # Select providers: GPU first, fallback to CPU
                providers = []
                if torch.cuda.is_available():
                    providers.append('CUDAExecutionProvider')
                    logger.info("CUDA GPU detected, using CUDAExecutionProvider")
                else:
                    logger.info("CUDA not available, using CPUExecutionProvider")

                providers.append('CPUExecutionProvider')

                # Load ONNX session
                self._yolo = ort.InferenceSession(YOLO_MODEL_PATH, providers=providers)

                # Log model details
                input_name = self._yolo.get_inputs()[0].name
                input_shape = self._yolo.get_inputs()[0].shape
                output_names = [o.name for o in self._yolo.get_outputs()]
                
                logger.info(f"ONNX YOLO loaded successfully in {time.time() - start_time:.2f}s")
                logger.debug(f"ONNX input: {input_name} {input_shape}")
                logger.debug(f"ONNX outputs: {output_names}")

            except Exception as e:
                logger.error(f"Failed to load YOLO ONNX model: {str(e)}", exc_info=True)
                raise

        return self._yolo

    @property
    def ocr(self):
        """Initialize EasyOCR reader"""
        if self._ocr is None:
            logger.info("Initializing EasyOCR reader...")
            start_time = time.time()

            try:
                # Validate languages
                supported_langs = ['en'] # ['ch_sim', 'fr', 'es', 'de']  # Add as needed
                valid_langs = [lang for lang in OCR_LANGUAGES if lang in supported_langs]
                if not valid_langs:
                    valid_langs = ['en']
                    logger.warning(f"No valid OCR languages specified, defaulting to ['en']")

                # Clean network name (remove quotes and spaces)
                clean_network_name = re.sub(r'[\"\']', '', CUSTOM_EASYOCR_NETWORK_NAME.strip())

                # Check if custom model directory exists
                if not os.path.exists(CUSTOM_EASYOCR_MODEL_DIR):
                    logger.warning(f"Custom OCR model directory not found: {CUSTOM_EASYOCR_MODEL_DIR}")
                    logger.info("Falling back to standard EasyOCR model")
                    self._ocr = easyocr.Reader(
                        lang_list=valid_langs,
                        gpu=torch.cuda.is_available(),
                        download_enabled=False
                    )
                else:
                    logger.info(f"Loading CUSTOM EasyOCR model: {CUSTOM_EASYOCR_NETWORK_NAME}")
                    logger.info(f"Model directory: {CUSTOM_EASYOCR_MODEL_DIR}")

                    # THIS IS THE KEY - MATCH YOUR WORKING CODE EXACTLY
                    self._ocr = easyocr.Reader(
                        lang_list=valid_langs,
                        model_storage_directory=CUSTOM_EASYOCR_MODEL_DIR,
                        recog_network=clean_network_name,  #CUSTOM_EASYOCR_NETWORK_NAME,
                        gpu=torch.cuda.is_available(),
                        download_enabled=False
                    )

                    logger.info(f"CUSTOM EasyOCR initialized in {time.time() - start_time:.2f}s")
                    logger.info(f"Using network: {CUSTOM_EASYOCR_NETWORK_NAME}")

                # Initialize reader
                # self._ocr = easyocr.Reader(
                #     lang_list=valid_langs,
                #     gpu=torch.cuda.is_available(),
                #     download_enabled=False  # Ensure models are pre-downloaded
                # )

                # # Custom EasyOCR Reader
                # self._ocr = easyocr.Reader(
                #     ['en'],
                #     model_storage_directory=CUSTOM_EASYOCR_MODEL_DIR, 
                #     recog_network='custom_example', # 'english_g2' # filename without extension
                #     gpu=torch.cuda.is_available(), #True  # set True if you have GPU
                # )

                # logger.info(f"EasyOCR initialized in {time.time() - start_time:.2f}s with languages: {valid_langs}")

            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {str(e)}", exc_info=True)
                raise

        return self._ocr
    
    
    @property
    def ner(self):
        """Load DistilBERT NER model and tokenizer"""
        if self._ner_model is None:
            logger.info("Loading DistilBERT NER model for key-value extraction...")
            start_time = time.time()

            try:
                # Check model path
                if not os.path.exists(NER_MODEL_PATH):
                    raise FileNotFoundError(f"NER model not found: {NER_MODEL_PATH}")

                # Load tokenizer and model
                self._ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
                self._ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)

                # Move to GPU if available
                if torch.cuda.is_available():
                    self._ner_model = self._ner_model.cuda()
                    logger.info("NER model moved to GPU")

                logger.info(f"NER model loaded in {time.time() - start_time:.2f}s")
                logger.debug(f"NER model: {NER_MODEL_PATH}")
                logger.debug(f"NER labels: {self._ner_model.config.id2label}")

            except Exception as e:
                logger.error(f"Failed to load NER model: {str(e)}", exc_info=True)
                raise

        return self._ner_tokenizer, self._ner_model


    @property
    def ner_pipeline(self):
        if self._ner_pipeline is None:
            logger.info("Initializing NER pipeline with aggregation strategy...")
            start_time = time.time()

            try:
                # Load model and tokenizer if not already loaded
                if self._ner_model is None or self._ner_tokenizer is None:
                    self._ner_tokenizer, self._ner_model = self.ner

                # Create pipeline with aggregation
                self._ner_pipeline = pipeline(
                    "token-classification",
                    model=self._ner_model,
                    tokenizer=self._ner_tokenizer,
                    aggregation_strategy="simple",  # Key parameter for grouping entities
                    device=0 if torch.cuda.is_available() else -1
                )
                
                logger.info(f"NER pipeline initialized in {time.time() - start_time:.2f}s")
                logger.debug(f"NER pipeline using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
                
            except Exception as e:
                logger.error(f"Failed to initialize NER pipeline: {str(e)}", exc_info=True)
                raise

        return self._ner_pipeline



model_manager = ModelManager()