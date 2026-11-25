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
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import THRESHOLD_PATH, anomaly_score, load_model, predict, select_threshold, train_model
from src.prediction import load_threshold, predict_single, save_threshold

app = Flask(__name__)
CORS(app)

# Lazy-load model
_model = None
_threshold = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def get_threshold():
    global _threshold
    if _threshold is None:
        _threshold = load_threshold()
    return _threshold


def _image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/health", methods=["GET"])
def health():
    """Model uptime check."""
    try:
        get_model()
        get_threshold()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
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

    try:
        with Image.open(io.BytesIO(file.read())) as uploaded_image:
            img = uploaded_image.convert("RGB")
        result = predict_single(
            img,
            model=get_model(),
            threshold=get_threshold(),
            include_images=True,
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
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    """
    Trigger retraining. Expects optional form field 'category' (default 'bottle').
    In production this would enqueue a background job (e.g., Celery/Prefect).
    """
    category = request.form.get("category", "bottle")
    try:
        from src.preprocessing import acquire_and_prepare

        # Re-acquire + preprocess
        X_train, X_test, y_test, _ = acquire_and_prepare(category, limit=300)

        # Train
        from sklearn.model_selection import train_test_split
        X_fit, X_validation = train_test_split(
            X_train,
            test_size=0.2,
            random_state=42,
        )
        model, _ = train_model(
            X_fit,
            epochs=10,
            validation_data=X_validation,
        )

        # New threshold
        scores_normal, _ = anomaly_score(model, X_validation)
        thr = select_threshold(scores_normal, percentile=90)
        save_threshold(thr)

        # Evaluate on test
        labels, scores = predict(model, X_test, thr)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        metrics = {
            "accuracy": float(accuracy_score(y_test, labels)),
            "precision": float(precision_score(y_test, labels, zero_division=0)),
            "recall": float(recall_score(y_test, labels, zero_division=0)),
            "f1": float(f1_score(y_test, labels, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, scores)),
        }
        global _model, _threshold
        _model = model
        _threshold = thr

        return jsonify({
            "status": "retrained",
            "category": category,
            "train_samples": int(len(X_train)),
            "fit_samples": int(len(X_fit)),
            "validation_samples": int(len(X_validation)),
            "test_samples": int(len(X_test)),
            "metrics": metrics,
        }), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/metrics", methods=["GET"])
def metrics_endpoint():
    """Return last known evaluation metrics (placeholder — recompute if needed)."""
    threshold = load_threshold() if os.path.exists(THRESHOLD_PATH) else None
    return jsonify({
        "model": "conv_autoencoder",
        "threshold": threshold,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
