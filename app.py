# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import from your modules
from src.utils import cleanup_temp_file
from src.validation import validate_image_file
from src.pipeline import process_receipt_pipeline
from src.model_manager import model_manager 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/receipts'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/process', methods=['POST'])
def process_receipt():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        validate_image_file(filepath)
        
        result = process_receipt_pipeline(
            image_path=filepath,
            yolo_model=model_manager.yolo,
            ocr_reader=model_manager.ocr,
            ner_tokenizer=model_manager.ner_tokenizer,
            ner_model=model_manager.ner_model
        )
        
        return jsonify({"status": "success", "result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        cleanup_temp_file(filepath)  # Ensure cleanup

# ... rest of your app.py ...

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

# --- Add this! ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)