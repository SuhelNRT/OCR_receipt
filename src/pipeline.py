# src/pipeline.py
from src.utils import step_logger, log_step
from PIL import Image
import os
from src.debug_saver import save_debug_artifacts
from config.settings import ENABLE_DEBUG_SAVE
from src.preprocessing import crop_receipt_with_yolo
from src.preprocessing import extract_text_with_ocr
from src.postprocessing import postprocess_kv_pairs
from transformers import AutoModelForTokenClassification
import torch

from src.debug_saver import save_debug_artifacts
from config.settings import ENABLE_DEBUG_SAVE

def process_receipt_pipeline(
    image_path: str,
    yolo_model,
    ocr_reader,
    ner_tokenizer,
    ner_model
) -> dict:
    result = {"raw_text": "", "entities": {}, "postprocessed": {}}
    filename = os.path.basename(image_path)

    with step_logger("load_image", filename=filename):
        image = Image.open(image_path)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

    with step_logger("crop_receipt", filename=filename):
        
        cropped_image = crop_receipt_with_yolo(image, yolo_model)

    with step_logger("ocr_extraction", filename=filename):
        
        raw_text = extract_text_with_ocr(cropped_image, ocr_reader)
        result["raw_text"] = raw_text

    with step_logger("ner_extraction", filename=filename):
        

        inputs = ner_tokenizer(
            raw_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )

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
        if current_key:
            entities[current_key] = " ".join(current_tokens)

        result["entities"] = entities

    with step_logger("postprocessing", filename=filename):
        
        final_kv = postprocess_kv_pairs(entities)
        result["postprocessed"] = final_kv

    # Final success log
    log_step("pipeline", "success", filename=filename, extracted_keys=list(final_kv.keys()))

    # 🔽 Optional: Save all artifacts
    if ENABLE_DEBUG_SAVE:
        save_debug_artifacts(
            original_image_path=image_path,
            cropped_image=cropped_image,
            raw_text=raw_text,
            kv_result=final_kv
        )

    return final_kv