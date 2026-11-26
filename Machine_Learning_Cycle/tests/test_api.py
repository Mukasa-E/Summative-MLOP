import io

import numpy as np
from PIL import Image

from api import app as api
from src import model


def _png_upload():
    image = Image.new("RGB", (8, 8), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_predict_rejects_missing_file():
    response = api.app.test_client().post("/predict")

    assert response.status_code == 400
    assert "No file provided" in response.get_json()["error"]


def test_predict_rejects_invalid_image():
    response = api.app.test_client().post(
        "/predict",
        data={"file": (io.BytesIO(b"not an image"), "invalid.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "not a valid" in response.get_json()["error"]


def test_predict_returns_class_and_visualizations(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda category: object())
    monkeypatch.setattr(api, "get_threshold", lambda category: 0.25)
    monkeypatch.setattr(
        api,
        "predict_single",
        lambda *args, **kwargs: {
            "label": 1,
            "score": 0.5,
            "threshold": 0.25,
            "anomaly": True,
            "confidence": 1.0,
            "reconstruction": np.zeros((224, 224, 3), dtype=np.float32),
            "error_map": np.zeros((224, 224), dtype=np.float32),
        },
    )

    response = api.app.test_client().post(
        "/predict",
        data={"file": (_png_upload(), "sample.png"), "category": "bottle"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = response.get_json()
    assert result["anomaly"] is True
    assert result["score"] == 0.5
    assert result["reconstruction_image"].startswith("data:image/png;base64,")
    assert result["error_heatmap"].startswith("data:image/png;base64,")
    assert result["heatmap_overlay"].startswith("data:image/png;base64,")


def test_retrain_rejects_unsupported_category():
    response = api.app.test_client().post("/retrain", data={"category": "../other"})

    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Unsupported MVTec category")


def test_retrain_starts_background_job_and_updates_loaded_model(monkeypatch):
    import src.preprocessing

    acquire_calls = []
    X_train = np.zeros((5, 224, 224, 3), dtype=np.float32)
    X_test = np.zeros((2, 224, 224, 3), dtype=np.float32)
    y_test = np.array([0, 1], dtype=np.int64)
    model = object()

    class ImmediateThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    def acquire(category, limit):
        acquire_calls.append((category, limit))
        return X_train, X_test, y_test, ["normal.png", "defect.png"]

    monkeypatch.setattr(src.preprocessing, "acquire_and_prepare", acquire)
    monkeypatch.setattr(api, "train_model", lambda *args, **kwargs: (model, []))
    monkeypatch.setattr(
        api,
        "anomaly_score",
        lambda trained_model, images: (np.array([0.1]), None),
    )
    monkeypatch.setattr(api, "select_threshold", lambda scores, percentile: 0.2)
    monkeypatch.setattr(
        api,
        "save_threshold",
        lambda threshold, category: None,
    )
    monkeypatch.setattr(
        api,
        "predict",
        lambda trained_model, images, threshold: (
            np.array([0, 1]),
            np.array([0.1, 0.3]),
        ),
    )
    monkeypatch.setattr(api.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(api, "_retrain_job", None)

    response = api.app.test_client().post("/retrain", data={"category": "bottle"})

    assert response.status_code == 202
    result = response.get_json()
    assert result["status"] == "running"
    status_response = api.app.test_client().get(
        f"/retrain/status/{result['job_id']}"
    )
    status = status_response.get_json()
    assert status_response.status_code == 200
    assert status["status"] == "completed"
    assert status["train_samples"] == 5
    assert status["test_samples"] == 2
    assert status["metrics"]["accuracy"] == 1.0
    assert acquire_calls == [("bottle", None)]
    assert api._models["bottle"] is model
    assert api._thresholds["bottle"] == 0.2


def test_retrain_rejects_second_active_job(monkeypatch):
    class PendingThread:
        def __init__(self, target, args, daemon):
            pass

        def start(self):
            pass

    monkeypatch.setattr(api.threading, "Thread", PendingThread)
    monkeypatch.setattr(api, "_retrain_job", None)
    first = api.app.test_client().post("/retrain", data={"category": "bottle"})
    second = api.app.test_client().post("/retrain", data={"category": "bottle"})
    api._retraining_lock.release()

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.get_json()["job_id"] == first.get_json()["job_id"]


def test_predict_accepts_supported_non_default_category(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda category: ("model", category))
    monkeypatch.setattr(api, "get_threshold", lambda category: 0.3)
    monkeypatch.setattr(
        api,
        "predict_single",
        lambda *args, **kwargs: {
            "label": 0,
            "score": 0.1,
            "threshold": 0.3,
            "anomaly": False,
            "confidence": 0.66,
            "reconstruction": np.zeros((224, 224, 3), dtype=np.float32),
            "error_map": np.zeros((224, 224), dtype=np.float32),
        },
    )

    response = api.app.test_client().post(
        "/predict",
        data={"file": (_png_upload(), "sample.png"), "category": "cable"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["anomaly"] is False


def test_predict_reports_missing_category_model(monkeypatch):
    def missing_model(category):
        raise FileNotFoundError(f"model for {category} is missing")

    monkeypatch.setattr(api, "get_model", missing_model)
    monkeypatch.setattr(api, "get_threshold", lambda category: 0.3)

    response = api.app.test_client().post(
        "/predict",
        data={"file": (_png_upload(), "sample.png"), "category": "cable"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    assert "Retrain this category first" in response.get_json()["error"]


def test_predict_rejects_unsupported_category():
    response = api.app.test_client().post(
        "/predict",
        data={"file": (_png_upload(), "sample.png"), "category": "../outside"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Unsupported MVTec category" in response.get_json()["error"]


def test_metrics_reports_untrained_category(monkeypatch):
    monkeypatch.setattr(api, "category_model_ready", lambda category: False)

    response = api.app.test_client().get("/metrics?category=cable")

    assert response.status_code == 200
    result = response.get_json()
    assert result["category"] == "cable"
    assert result["model_ready"] is False
    assert result["threshold"] is None


def test_category_model_ready_requires_model_and_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "MODEL_DIR", str(tmp_path))

    assert model.category_model_ready("cable") is False
    (tmp_path / "autoencoder_mvtec_cable.pth").write_bytes(b"model")
    assert model.category_model_ready("cable") is False
    (tmp_path / "threshold_cable.npy").write_bytes(b"threshold")
    assert model.category_model_ready("cable") is True


def test_select_threshold_validates_percentile_and_scores():
    assert np.isclose(model.select_threshold([1, 2, 3, 4], percentile=70), 3.1)

    with np.testing.assert_raises(ValueError):
        model.select_threshold([], percentile=70)
    with np.testing.assert_raises(ValueError):
        model.select_threshold([1, 2], percentile=101)
