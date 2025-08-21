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
from src.postprocessing import organise_ner_result
from src.preprocessing import find_best_resize_factor_adaptive


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
    # ner_pipeline = model_manager.ner_pipeline
    
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

            raw_text = ''

            # ocr_results = ocr_reader.readtext(cropped_image)

            # raw_text = "\n".join([text for (bbox, text, prob) in ocr_results ])
            
            raw_text, resized_factor = find_best_resize_factor_adaptive(cropped_image, ocr_reader)
            
            artifacts["raw_text"] = raw_text
            log_step("ocr_extraction", "completed", 
                    filename=filename, 
                    resized_factor=resized_factor,
                    char_count=len(raw_text))
            

        with step_logger("ner_extraction", filename=filename):
            if not raw_text.strip():
                raise ValueError("No text extracted from image")
            
            try:
                final_entities = dict()
                ner_pipeline = model_manager.ner_pipeline
                ner_results = ner_pipeline(raw_text)

                final_entities = organise_ner_result(ner_results)
                
                artifacts["ner_entities"] = final_entities
                log_step("ner_extraction", "completed", 
                        filename=filename, 
                        num_items=len(final_entities.get("items", [])),
                        shop_name=final_entities.get("shop", {}).get("name", "unknown"))
                

            except Exception as e:
                log_step("ner_extraction", "failed", 
                        filename=filename, 
                        error=str(e), 
                        error_type=type(e).__name__)
                raise
            

        with step_logger("postprocessing", filename=filename):
            # final_kv = postprocess_kv_pairs(entities)
            final_kv = final_entities
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