"""
Flask REST API for MVTec AD Anomaly Detection.
Endpoints:
  GET  /health      - model uptime / health check
  POST /predict     - upload one image -> anomaly prediction
  POST /retrain     - trigger retraining pipeline
  GET  /metrics     - basic evaluation metrics
"""

import os
import io
import base64
import threading
import uuid
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import (
    DEFAULT_CATEGORY,
    DEFAULT_THRESHOLD_PERCENTILE,
    anomaly_score,
    category_model_ready,
    load_model,
    load_threshold as load_saved_threshold,
    predict,
    select_threshold,
    train_model,
)
from src.prediction import load_threshold, predict_single, save_threshold
from src.preprocessing import MVTEC_CATEGORIES

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Lazy-load model
_models = {}
_thresholds = {}
_retraining_lock = threading.Lock()
_retrain_job_lock = threading.Lock()
_retrain_job = None


def get_model(category=DEFAULT_CATEGORY):
    if category not in MVTEC_CATEGORIES:
        raise ValueError(f"Unsupported MVTec category: {category}")
    if category not in _models:
        _models[category] = load_model(category=category)
    return _models[category]


def get_threshold(category=DEFAULT_CATEGORY):
    if category not in MVTEC_CATEGORIES:
        raise ValueError(f"Unsupported MVTec category: {category}")
    if category not in _thresholds:
        _thresholds[category] = load_threshold(category=category)
    return _thresholds[category]


def _image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/health", methods=["GET"])
def health():
    """Model uptime check."""
    try:
        get_model(DEFAULT_CATEGORY)
        get_threshold(DEFAULT_CATEGORY)
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "default_category": DEFAULT_CATEGORY,
        }), 200
    except Exception as exc:
        return jsonify({"status": "unhealthy", "error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """Upload a single image and get anomaly prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use form field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    category = request.form.get("category", DEFAULT_CATEGORY)
    if category not in MVTEC_CATEGORIES:
        return jsonify({
            "error": f"Unsupported MVTec category '{category}'. "
            f"Choose from: {', '.join(MVTEC_CATEGORIES)}"
        }), 400

    try:
        with Image.open(io.BytesIO(file.read())) as uploaded_image:
            img = uploaded_image.convert("RGB")
        result = predict_single(
            img,
            model=get_model(category),
            threshold=get_threshold(category),
            include_images=True,
            category=category,
        )
        reconstruction = result.pop("reconstruction")
        error_map = result.pop("error_map")

        reconstruction_image = Image.fromarray(
            np.asarray(np.clip(reconstruction * 255, 0, 255), dtype=np.uint8)
        )
        error_max = float(error_map.max())
        normalized_error = error_map / error_max if error_max > 0 else error_map
        error_image = Image.fromarray(
            np.asarray(np.clip(normalized_error * 255, 0, 255), dtype=np.uint8)
        )
        heatmap_image = ImageOps.colorize(
            error_image,
            black="#111827",
            mid="#f59e0b",
            white="#ef4444",
        )
        original_image = img.resize(reconstruction_image.size, Image.Resampling.LANCZOS)
        overlay_image = Image.blend(original_image, heatmap_image, alpha=0.45)

        result.update({
            "filename": file.filename,
            "reconstruction_image": _image_data_url(reconstruction_image),
            "error_heatmap": _image_data_url(heatmap_image),
            "heatmap_overlay": _image_data_url(overlay_image),
        })
        return jsonify(result), 200

    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid supported image."}), 400
    except FileNotFoundError:
        return jsonify({
            "error": f"No trained model is available for '{category}'. "
            "Retrain this category first."
        }), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _perform_retraining(category):
    from src.preprocessing import acquire_and_prepare
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    X_train, X_test, y_test, _ = acquire_and_prepare(category, limit=None)
    X_fit, X_validation = train_test_split(
        X_train,
        test_size=0.2,
        random_state=42,
    )
    model, _ = train_model(
        X_fit,
        epochs=10,
        validation_data=X_validation,
        category=category,
    )

    scores_normal, _ = anomaly_score(model, X_validation)
    threshold = select_threshold(
        scores_normal,
        percentile=DEFAULT_THRESHOLD_PERCENTILE,
    )
    labels, scores = predict(model, X_test, threshold)
    metrics = {
        "accuracy": float(accuracy_score(y_test, labels)),
        "precision": float(precision_score(y_test, labels, zero_division=0)),
        "recall": float(recall_score(y_test, labels, zero_division=0)),
        "f1": float(f1_score(y_test, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
    }

    save_threshold(threshold, category=category)
    _models[category] = model
    _thresholds[category] = threshold
    return {
        "status": "completed",
        "category": category,
        "train_samples": int(len(X_train)),
        "fit_samples": int(len(X_fit)),
        "validation_samples": int(len(X_validation)),
        "test_samples": int(len(X_test)),
        "metrics": metrics,
    }


def _run_retraining_job(job_id, category):
    global _retrain_job
    try:
        result = _perform_retraining(category)
        with _retrain_job_lock:
            if _retrain_job and _retrain_job["job_id"] == job_id:
                _retrain_job.update(result)
    except Exception as exc:
        with _retrain_job_lock:
            if _retrain_job and _retrain_job["job_id"] == job_id:
                _retrain_job.update({"status": "failed", "error": str(exc)})
    finally:
        _retraining_lock.release()


@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    """Start a retraining job for the supported model category."""
    global _retrain_job
    category = request.form.get("category", DEFAULT_CATEGORY)
    if category not in MVTEC_CATEGORIES:
        return jsonify({
            "error": f"Unsupported MVTec category '{category}'. Choose from: "
            f"{', '.join(MVTEC_CATEGORIES)}"
        }), 400

    if not _retraining_lock.acquire(blocking=False):
        with _retrain_job_lock:
            return jsonify({
                "error": "A retraining job is already running.",
                "job_id": _retrain_job["job_id"] if _retrain_job else None,
            }), 409

    job_id = uuid.uuid4().hex
    with _retrain_job_lock:
        _retrain_job = {
            "job_id": job_id,
            "status": "running",
            "category": category,
        }

    worker = threading.Thread(
        target=_run_retraining_job,
        args=(job_id, category),
        daemon=True,
    )
    try:
        worker.start()
    except Exception as exc:
        with _retrain_job_lock:
            _retrain_job.update({"status": "failed", "error": str(exc)})
        _retraining_lock.release()
        return jsonify({"error": f"Could not start retraining: {exc}"}), 500

    return jsonify({
        "status": "running",
        "job_id": job_id,
        "category": category,
    }), 202


@app.route("/retrain/status/<job_id>", methods=["GET"])
def retrain_status_endpoint(job_id):
    with _retrain_job_lock:
        if not _retrain_job or _retrain_job["job_id"] != job_id:
            return jsonify({"error": "Retraining job not found."}), 404
        return jsonify(dict(_retrain_job)), 200


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Uploaded image exceeds the 10 MB request limit."}), 413


@app.route("/metrics", methods=["GET"])
def metrics_endpoint():
    """Return last known evaluation metrics (placeholder — recompute if needed)."""
    category = request.args.get("category", DEFAULT_CATEGORY)
    if category not in MVTEC_CATEGORIES:
        return jsonify({"error": f"Unsupported MVTec category: {category}"}), 400
    try:
        threshold = load_saved_threshold(category=category)
    except FileNotFoundError:
        threshold = None
    return jsonify({
        "model": "conv_autoencoder",
        "category": category,
        "model_ready": category_model_ready(category),
        "threshold": threshold,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
