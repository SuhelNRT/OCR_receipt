# src/pipeline.py
from src.utils import step_logger, log_step
import cv2
import os
from src.debug_saver import save_debug_artifacts
from config.settings import ENABLE_DEBUG_SAVE
from src.postprocessing import postprocess_kv_pairs
from transformers import AutoModelForTokenClassification
import torch
from src.preprocessing import preprocess_image, postprocess_output, preprocess_image_new, postprocess_output_new, xywh2xyxy
from src.debug_saver import save_debug_artifacts
from config.settings import ENABLE_DEBUG_SAVE, YOLO_ONNX_INPUT_SIZE, YOLO_ONNX_CONF_THRESHOLD
import numpy as np
from src.model_manager import model_manager
from contextlib import contextmanager


@contextmanager
def managed_image(image_path: str):
    """Context manager to safely load and close image"""
    img = None
    try:
        # img = Image.open(path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image (corrupted or invalid): {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        log_step("image_load", "completed", 
                filename=os.path.basename(image_path), 
                shape=f"{img.shape[1]}x{img.shape[0]}", 
                channels=img.shape[2])
        
        yield img, gray

    except Exception as e:
        log_step("image_load", "failed", error=str(e))
        raise
    finally:
        pass



def process_receipt_pipeline(image_path: str) -> dict:

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    filename = os.path.basename(image_path)
    log_step("pipeline_start", "started", filename=filename)

    # Initialize result container
    artifacts = {
        "raw_text": "",
        "ner_entities": {},
        "final_output": {}
    }

    # Get models from manager
    yolo_session = model_manager.yolo
    ocr_reader = model_manager.ocr
    ner_tokenizer, ner_model = model_manager.ner
    
    try :
        with step_logger("load_image", filename=filename):
            with managed_image(image_path) as (color_image, gray_image):
                # orig_size = (image.width, image.height)
                orig_h, orig_w = gray_image.shape[:2]
                color_copy = color_image.copy()
                gray_copy = gray_image.copy()
                img_size = YOLO_ONNX_INPUT_SIZE


        with step_logger("crop_receipt", filename=filename):

            input_tensor = preprocess_image_new(color_copy, img_size)

            # Run ONNX inference
            input_name = yolo_session.get_inputs()[0].name
            outputs = yolo_session.run(None, {input_name: input_tensor})

            # changes
            detection_output = outputs[0]

            #Handle different output format
            if detection_output.ndim == 3:
                det = detection_output[0].transpose(1, 0)
            else:
                det = detection_output.transpose(1, 0)

            # det = detection_output[0].transpose(1,0) 
            best_box = postprocess_output_new(det, YOLO_ONNX_CONF_THRESHOLD)

            if best_box is None:
                raise ValueError("No receipt detected with sufficient confidence")
            
            print("best box is: " , best_box)
            boxes = xywh2xyxy(best_box, orig_w, orig_h, img_size)
            x1, y1, x2, y2 = boxes[0]
    
            # Clip to image boundaries
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))

            cropped_image = gray_copy[y1:y2, x1:x2]

            log_step("crop_receipt", "completed", 
                    filename=filename, 
                    crop_box=[int(x1), int(y1), int(x2), int(y2)]),
                    # confidence=scores[largest_idx] if scores else None)
        

        with step_logger("ocr_extraction", filename=filename):
            # img_np = np.array(cropped_image)
            ocr_results = ocr_reader.readtext(
                cropped_image,
                detail=1)
                # paragraph=False,
                # min_size=8,
                # text_threshold=0.7,
                # low_text=0.4,
                # contrast_ths=0.1,
                # adjust_contrast=0.5

            # Sort by vertical position then horizontal
            ocr_results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

            # Group by lines
            lines = {}
            for (bbox, text, prob) in ocr_results:
                y_pos = bbox[0][1]
                line_key = round(y_pos / 15)  # Group every ~15px vertically
                lines.setdefault(line_key, []).append({
                    "text": text,
                    "confidence": prob
                })

            # Join text
            raw_text = "\n".join([
                " ".join([item["text"] for item in items]) 
                for k in sorted(lines) for items in [lines[k]]
            ])
            
            artifacts["raw_text"] = raw_text
            log_step("ocr_extraction", "completed", 
                    filename=filename, 
                    num_lines=len(lines),
                    char_count=len(raw_text))
            

        with step_logger("ner_extraction", filename=filename):
            if not raw_text.strip():
                raise ValueError("No text extracted from image")
            
            inputs = ner_tokenizer(
                raw_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
                # return_offsets_mapping = True
            )

            # Move to GPU if available
            device = next(ner_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = ner_model(**inputs)

            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()
            input_ids = inputs["input_ids"].cpu().numpy()[0]
            # offsets = inputs.get("offset_mapping", None)

            # Convert to tokens and labels
            tokens = ner_tokenizer.convert_ids_to_tokens(input_ids)
            labels = [ner_model.config.id2label[p] for p in predictions[0]]

            entities = {}
            current_key = None
            current_tokens = []
            # current_start = None

            for i, (token, label) in enumerate(zip(tokens, labels)):
                if label.startswith("B-"):
                    # Save previous entity
                    if current_key and current_tokens:
                        value = " ".join(current_tokens).strip()
                        if value:
                            entities[current_key] = value
                    
                    # Start new entity
                    current_key = label[2:]  # Remove "B-" prefix
                    current_tokens = [token.replace("##", "")]
                    # current_start = i
                
                elif label.startswith("I-") and current_key == label[2:]:
                    current_tokens.append(token.replace("##", ""))
                
                else:
                    # End current entity
                    if current_key and current_tokens:
                        value = " ".join(current_tokens).strip()
                        if value:
                            entities[current_key] = value
                        current_key = None
                        current_tokens = []
            
            # Final entity
            if current_key and current_tokens:
                value = " ".join(current_tokens).strip()
                if value:
                    entities[current_key] = value
            
            artifacts["ner_entities"] = entities
            log_step("ner_extraction", "completed", 
                    filename=filename, 
                    num_entities=len(entities))


        with step_logger("postprocessing", filename=filename):
            # final_kv = postprocess_kv_pairs(entities)
            final_kv = entities
            artifacts["final_output"] = final_kv
            log_step("postprocessing", "completed", 
                        filename=filename, 
                        extracted_keys=list(final_kv.keys()))

        # 🔽 Optional: Save all artifacts
        if ENABLE_DEBUG_SAVE:
            with step_logger("debug_save", filename=filename):
                save_debug_artifacts(
                    original_image_path=image_path,
                    cropped_image=cropped_image,
                    raw_text=raw_text,
                    kv_result=final_kv
                )

        # Final success log
        log_step("pipeline", "success", 
                filename=filename, 
                extracted_keys=list(final_kv.keys()))

        return final_kv
    
    except Exception as e:
        log_step("pipeline", "failed", 
                filename=filename, 
                error=str(e), 
                error_type=type(e).__name__)
        raise