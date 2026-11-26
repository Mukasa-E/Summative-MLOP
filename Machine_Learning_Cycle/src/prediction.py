"""
Single-image prediction entry point.
Used by the API and the UI to classify one image as normal or anomalous.
"""

import numpy as np
from PIL import Image

from .preprocessing import preprocess_image, IMG_SIZE
from .model import (
    anomaly_score,
    load_model,
    load_threshold as _load_threshold,
    save_threshold as _save_threshold,
)
from .preprocessing import MVTEC_CATEGORIES


def load_threshold(category="bottle"):
    """Load the saved anomaly threshold."""
    if category not in MVTEC_CATEGORIES:
        raise ValueError(f"Unsupported MVTec category: {category}")
    return _load_threshold(category=category)


def save_threshold(value, category="bottle"):
    """Save the anomaly threshold."""
    if category not in MVTEC_CATEGORIES:
        raise ValueError(f"Unsupported MVTec category: {category}")
    _save_threshold(value, category=category)


def predict_single(
    image_path_or_pil,
    model=None,
    threshold=None,
    include_images=False,
    category="bottle",
):
    """
    Predict whether a single image is anomalous.

    Parameters
    ----------
    image_path_or_pil : str, os.PathLike, or PIL.Image.Image
    model : optional
        Reuse a loaded model when serving repeated API requests.
    threshold : float, optional
        Reuse a loaded threshold when serving repeated API requests.
    include_images : bool
        Include the reconstruction and per-pixel error map in the result.

    Returns
    -------
    dict with keys:
        label        : 0 (normal) or 1 (anomalous)
        score        : reconstruction error (float)
        anomaly      : bool
        confidence   : distance from threshold (0-1)
        reconstruction and error_map : arrays, only when include_images=True
    """
    if model is None:
        model = load_model(category=category)
    if threshold is None:
        threshold = load_threshold(category=category)

    if isinstance(image_path_or_pil, (str, bytes)) or hasattr(image_path_or_pil, "__fspath__"):
        with Image.open(image_path_or_pil) as image:
            img = image.convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    arr = preprocess_image(img, IMG_SIZE)
    X = np.expand_dims(arr, axis=0)

    scores, reconstructions = anomaly_score(model, X)
    score = float(scores[0])
    label = int(score > threshold)

    # Simple confidence: how far the score is from the threshold
    confidence = float(min(abs(score - threshold) / max(threshold, 1e-6), 1.0))

    result = {
        "label": label,
        "score": score,
        "threshold": float(threshold),
        "anomaly": bool(label == 1),
        "confidence": confidence,
    }
    if include_images:
        reconstruction = np.transpose(reconstructions[0], (1, 2, 0))
        error_map = np.mean(np.square(arr - reconstruction), axis=2)
        result["reconstruction"] = np.clip(reconstruction, 0.0, 1.0)
        result["error_map"] = error_map
    return result


def predict_batch(image_paths, category="bottle"):
    """Predict for a list of image paths. Returns list of result dicts."""
    return [predict_single(p, category=category) for p in image_paths]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = predict_single(sys.argv[1])
        print(res)
