# src/pipeline.py
from src.preprocessing import crop_receipt_with_yolo, extract_text_with_ocr
from src.postprocessing import postprocess_kv_pairs
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from src.debug_saver import save_debug_artifacts
from config.settings import ENABLE_DEBUG_SAVE

from PIL import Image
from src.utils import load_image_from_path
from src.preprocessing import extract_text_with_ocr
from src.postprocessing import postprocess_kv_pairs

def process_receipt_pipeline(
    image_path: str,
    yolo_model,
    ocr_reader,
    ner_tokenizer,
    ner_model
) -> dict:
    

    # 1. Load image
    image = load_image_from_path(image_path)
    
    # 2. Crop receipt
    cropped_image = crop_receipt_with_yolo(image, yolo_model)
    
    # 3. OCR
    raw_text = extract_text_with_ocr(cropped_image, ocr_reader)
    
    # 4. NER to key-value
    inputs = ner_tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = ner_model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [ner_model.config.id2label[p.item()] for p in predictions[0]]
    
    entities = {}
    current_key = None
    current_tokens = []
    
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_key:
                entities[current_key] = " ".join(current_tokens)
            current_key = label[2:]
            current_tokens = [token.replace("##", "")]
        elif label.startswith("I-") and current_key == label[2:]:
            current_tokens.append(token.replace("##", ""))
        else:
            if current_key:
                entities[current_key] = " ".join(current_tokens)
                current_key = None
                current_tokens = []
    
    # 5. Post-process
    final_kv = postprocess_kv_pairs(entities)

     # 🔽 Optional: Save all artifacts
    if ENABLE_DEBUG_SAVE:
        save_debug_artifacts(
            original_image_path=image_path,
            cropped_image=cropped_image,
            raw_text=raw_text,
            kv_result=final_kv
        )

    return final_kv