"""Features:
  - Model prediction (upload one image)
  - Visualizations (reconstruction, error heatmap, feature insights)
  - Bulk data upload for retraining
  - Trigger retraining button
  - Model uptime / health display
"""

import os
import sys
import base64
import io
import time
import uuid
import streamlit as st
from PIL import Image
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import MVTEC_CATEGORIES
from src.model import DEFAULT_CATEGORY, DEFAULT_THRESHOLD_PERCENTILE

API_URL = os.getenv("API_URL", "http://localhost:5000")
MAX_TRAINING_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="MVTec AD-Anomaly Detection", page_icon="🔍", layout="wide")

st.title("MVTec AD Manufacturing Defect Detection")
st.markdown("""
Anomaly detection on the **MVTec AD** dataset. The model learns what *normal* looks like
and flags deviations as defects (scratches, dents, missing components).
""")

#Sidebar

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Visualizations", "Upload & Retrain", "Model Uptime"])

st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
try:
    import requests
    resp = requests.get(f"{API_URL}/health", timeout=3)
    if resp.ok:
        st.sidebar.success("API online")
    else:
        st.sidebar.error("API error")
except Exception:
    st.sidebar.warning("API not reachable (start `api/app.py` or docker-compose)")


#helpers

def api_predict(image_file, category=DEFAULT_CATEGORY):
    """Call /predict endpoint for a selected MVTec product category."""
    import requests
    files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
    r = requests.post(
        f"{API_URL}/predict",
        files=files,
        data={"category": category},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def api_retrain(category="bottle"):
    """Start a background retraining job."""
    import requests
    r = requests.post(f"{API_URL}/retrain", data={"category": category}, timeout=15)
    r.raise_for_status()
    return r.json()


def api_retrain_status(job_id):
    """Read the status of a background retraining job."""
    import requests
    r = requests.get(f"{API_URL}/retrain/status/{job_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_metrics(category=DEFAULT_CATEGORY):
    import requests
    r = requests.get(
        f"{API_URL}/metrics",
        params={"category": category},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def display_prediction_images(result):
    """Display decoded images returned by the prediction API."""
    captions = [
        ("reconstruction_image", "Model reconstruction"),
        ("error_heatmap", "Pixel error heatmap"),
        ("heatmap_overlay", "Error heatmap overlay"),
    ]
    columns = st.columns(len(captions))
    for column, (key, caption) in zip(columns, captions):
        image_data = result.get(key)
        if image_data:
            encoded = image_data.split(",", 1)[-1]
            column.image(base64.b64decode(encoded), caption=caption, width="stretch")


# Page:predict

if page == "Predict":
    st.header("Model Prediction - Single Image")
    st.write("Upload one image (PNG/JPG). The model returns normal vs anomalous + anomaly score.")

    prediction_category = st.selectbox(
        "Product category",
        MVTEC_CATEGORIES,
        index=MVTEC_CATEGORIES.index(DEFAULT_CATEGORY),
        key="prediction_category",
    )
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", width='stretch')

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    result = api_predict(uploaded, prediction_category)
                    st.subheader("Result")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prediction", "ANOMALOUS" if result["anomaly"] else "NORMAL")
                    col2.metric("Anomaly Score", f"{result['score']:.6f}")
                    col3.metric("Threshold", f"{result['threshold']:.6f}")
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                    display_prediction_images(result)
                    st.json({
                        key: value
                        for key, value in result.items()
                        if key not in {
                            "reconstruction_image",
                            "error_heatmap",
                            "heatmap_overlay",
                        }
                    })
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


#Page:Visualizations

elif page == "Visualizations":
    st.header("Data Visualizations & Feature Insights")
    st.write(
        "Upload a real image to inspect its reconstruction, per-pixel error map, "
        "and overlay from the trained model."
    )
    visualization_category = st.selectbox(
        "Product category",
        MVTEC_CATEGORIES,
        index=MVTEC_CATEGORIES.index(DEFAULT_CATEGORY),
        key="visualization_category",
    )
    uploaded_viz = st.file_uploader(
        "Choose an image for model-based visualizations",
        type=["png", "jpg", "jpeg", "bmp", "tif"],
        key="visualization_upload",
    )
    if uploaded_viz is not None:
        st.image(uploaded_viz, caption="Input image", width='stretch')
        if st.button("Generate model visualizations", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    result = api_predict(uploaded_viz, visualization_category)
                    st.subheader(
                        "Anomalous" if result["anomaly"] else "Normal"
                    )
                    st.write(
                        f"Reconstruction error: {result['score']:.6f} "
                        f"(decision threshold: {result['threshold']:.6f})"
                    )
                    display_prediction_images(result)
                except Exception as exc:
                    st.error(f"Visualization failed: {exc}")


# Page:Upload and Retrain

elif page == "Upload & Retrain":
    st.header("📤 Bulk Data Upload & Retraining")
    st.write(
        "Select the product category and upload verified defect-free images only. "
        "Every saved image is treated as normal training data; defective images "
        "must not be uploaded here."
    )

    category = st.selectbox(
        "MVTec AD category",
        MVTEC_CATEGORIES,
        index=MVTEC_CATEGORIES.index(DEFAULT_CATEGORY),
        key="upload_category",
    )

    uploaded_files = st.file_uploader("Choose multiple images", type=["png", "jpg", "jpeg", "bmp", "tif"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected for {category}.")
        save_dir = os.path.join(PROJECT_ROOT, "data", "uploads", category, "normal")
        oversized_uploads = [
            uploaded_file.name
            for uploaded_file in uploaded_files
            if uploaded_file.size > MAX_TRAINING_UPLOAD_BYTES
        ]
        if oversized_uploads:
            st.error("Each training image must be 10 MB or smaller.")
        confirmed_normal = st.checkbox(
            f"I confirm every selected image is a defect-free normal {category}.",
            key="confirm_normal_uploads",
        )
        if st.button(
            "Save uploaded data for retraining",
            disabled=not confirmed_normal or bool(oversized_uploads),
        ):
            try:
                images_to_save = []
                for uploaded_file in uploaded_files:
                    safe_name = secure_filename(uploaded_file.name)
                    if not safe_name:
                        raise ValueError("An uploaded image has an invalid filename.")
                    image = Image.open(
                        io.BytesIO(uploaded_file.getvalue())
                    ).convert("RGB")
                    images_to_save.append((safe_name, image))

                os.makedirs(save_dir, exist_ok=True)
                for safe_name, image in images_to_save:
                    filename = f"{uuid.uuid4().hex}_{safe_name}"
                    image.save(os.path.join(save_dir, filename))
                st.success(
                    f"Saved {len(uploaded_files)} verified normal image(s) for retraining."
                )
            except Exception as exc:
                st.error(f"Could not save uploaded images: {exc}")

    st.markdown("---")
    st.subheader("Trigger Retraining")
    st.write(
        f"Retraining uses the original {category} training images plus images saved "
        f"above under `data/uploads/{category}/normal/`, then recalculates this category's threshold. "
        f"The threshold is calibrated at the p{DEFAULT_THRESHOLD_PERCENTILE} percentile "
        "of held-out normal validation scores. Training may take several minutes; "
        "progress is checked without blocking prediction or health requests."
    )
    if st.button("🔁 Trigger Retraining", type="primary"):
        with st.spinner("Retraining model in the background..."):
            try:
                job = api_retrain(category)
                while job["status"] == "running":
                    time.sleep(2)
                    job = api_retrain_status(job["job_id"])
                if job["status"] == "completed":
                    st.success("Retraining complete!")
                    st.json(job)
                else:
                    st.error(f"Retraining failed: {job.get('error', 'Unknown error')}")
            except Exception as exc:
                st.error(f"Retraining failed: {exc}")

# Page: Model Uptime

elif page == "Model Uptime":
    st.header("Model Uptime & Health")
    st.write("Monitor whether the prediction API is online and responsive.")

    if st.button("Check Health"):
        try:
            import requests
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.json(r.json())
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

    st.markdown("---")
    st.subheader("Current Metrics")
    try:
        selected_metrics_category = st.selectbox(
            "Metrics category",
            MVTEC_CATEGORIES,
            index=MVTEC_CATEGORIES.index(DEFAULT_CATEGORY),
            key="metrics_category",
        )
        m = api_metrics(selected_metrics_category)
        if not m.get("model_ready"):
            st.info(
                f"The {selected_metrics_category} model has not been trained yet. "
                "Choose this category in Upload & Retrain first."
            )
        st.json(m)
    except Exception as exc:
        st.warning(f"Could not fetch metrics: {exc}")
