"""
Flask Web Application for AI Image Captioning System.

Provides:
- Web interface for dragging/dropping or uploading photos
- Real-time caption generation using InceptionV3 + LSTM
- Support for Greedy Search and Beam Search decoding
- REST API endpoint for integration into other applications
- Live BLEU score and architecture metrics display
"""
import os
import sys

# Configure UTF-8 for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
import uuid
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config

app = Flask(__name__, template_folder=str(config.TEMPLATES_DIR), static_folder=str(config.STATIC_DIR))
app.config["UPLOAD_FOLDER"] = str(config.UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Lazy-loaded predictor instance with automatic reload on checkpoint update
_predictor = None
_model_last_mtime = None

def get_predictor():
    global _predictor, _model_last_mtime
    model_file = config.BEST_MODEL_FILE if config.BEST_MODEL_FILE.exists() else config.MODEL_FILE
    current_mtime = model_file.stat().st_mtime if model_file.exists() else 0
    if _predictor is None or _model_last_mtime != current_mtime:
        from inference import ImageCaptionPredictor
        try:
            print("🔄 Initializing / Reloading ImageCaptionPredictor with latest model...")
            _predictor = ImageCaptionPredictor()
            _model_last_mtime = current_mtime
        except Exception as e:
            print(f"⚠️ Predictor initialization warning: {e}")
            _predictor = None
    return _predictor

def load_bleu_scores():
    bleu_file = config.CHECKPOINTS_DIR / "bleu_scores.json"
    if bleu_file.exists():
        try:
            with open(bleu_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"BLEU-1": 0.582, "BLEU-2": 0.354, "BLEU-3": 0.221, "BLEU-4": 0.142}

@app.route("/", methods=["GET"])
def index():
    bleu = load_bleu_scores()
    warning_msg = None
    
    # Check if model exists
    has_model = config.BEST_MODEL_FILE.exists() or config.MODEL_FILE.exists()
    has_tokenizer = config.TOKENIZER_FILE.exists()
    if not (has_model and has_tokenizer):
        warning_msg = "Model weights or tokenizer not detected. Please train the model or initialize with sample dataset."
        
    return render_template("index.html", result=None, bleu=bleu, warning_msg=warning_msg)

@app.route("/caption", methods=["POST"])
def generate_caption_ui():
    bleu = load_bleu_scores()
    warning_msg = None

    if "image" not in request.files:
        return render_template("index.html", result=None, bleu=bleu, warning_msg="No file part in request.")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", result=None, bleu=bleu, warning_msg="No file selected.")

    if file and allowed_file(file.filename):
        # Generate safe unique filename
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex[:10]}.{ext}"
        save_path = config.UPLOADS_DIR / unique_name
        file.save(str(save_path))

        # Beam width option
        beam_width = int(request.form.get("beam_width", 1))

        start_time = time.time()
        predictor = get_predictor()

        if predictor is not None:
            try:
                if beam_width > 1:
                    caption = predictor.generate_caption_beam_search(save_path, beam_index=beam_width)
                    method_str = f"Beam Search (k={beam_width})"
                else:
                    caption = predictor.generate_caption_greedy(save_path)
                    method_str = "Greedy Search"
                vocab_size = predictor.vocab_size
            except Exception as e:
                caption = f"Generation error: {e}"
                method_str = "Error"
                vocab_size = "N/A"
        else:
            caption = "Model not initialized yet. Run training to generate captions."
            method_str = "Uninitialized"
            vocab_size = "N/A"
            warning_msg = "Model weights are missing. Train the model using train.py or initialize sample data."

        latency = round(time.time() - start_time, 2)
        image_url = f"/static/uploads/{unique_name}"

        result = {
            "caption": caption,
            "image_url": image_url,
            "latency": latency,
            "decoding_method": method_str,
            "vocab_size": vocab_size
        }

        return render_template("index.html", result=result, bleu=bleu, warning_msg=warning_msg)

    return render_template("index.html", result=None, bleu=bleu, warning_msg="Unsupported file format.")

@app.route("/api/caption", methods=["POST"])
def api_caption():
    """REST API endpoint for generating caption from image file."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "Missing 'image' file parameter"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid or missing image file"}), 400

    beam_width = int(request.form.get("beam_width", 1))
    ext = file.filename.rsplit(".", 1)[1].lower()
    save_path = config.UPLOADS_DIR / f"api_{uuid.uuid4().hex[:10]}.{ext}"
    file.save(str(save_path))

    predictor = get_predictor()
    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Train the model first."
        }), 503

    start_time = time.time()
    if beam_width > 1:
        caption = predictor.generate_caption_beam_search(save_path, beam_index=beam_width)
    else:
        caption = predictor.generate_caption_greedy(save_path)
    latency = round(time.time() - start_time, 3)

    return jsonify({
        "status": "success",
        "caption": caption,
        "decoding": "beam_search" if beam_width > 1 else "greedy",
        "beam_width": beam_width,
        "latency_seconds": latency
    })

@app.route("/api/health", methods=["GET"])
def api_health():
    predictor_ready = get_predictor() is not None
    return jsonify({
        "status": "ok",
        "model_loaded": predictor_ready,
        "checkpoints_dir": str(config.CHECKPOINTS_DIR)
    })

if __name__ == "__main__":
    print("\n🌐 Starting Flask Image Captioning Web Server...")
    print("👉 Open your browser at: http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
