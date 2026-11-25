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
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

API_URL = os.getenv("API_URL", "http://localhost:5000")

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

def api_predict(image_file):
    """Call /predict endpoint."""
    import requests
    files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
    r = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def api_retrain(category="bottle"):
    """Call /retrain endpoint."""
    import requests
    r = requests.post(f"{API_URL}/retrain", data={"category": category}, timeout=300)
    r.raise_for_status()
    return r.json()


def api_metrics():
    import requests
    r = requests.get(f"{API_URL}/metrics", timeout=10)
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
    st.header("Model Prediction -Single Image")
    st.write("Upload one image (PNG/JPG). The model returns normal vs anomalous + anomaly score.")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", width='stretch')

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    result = api_predict(uploaded)
                    st.subheader("Result")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prediction", "🚨 ANOMALOUS" if result["anomaly"] else "✅ NORMAL")
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
                    result = api_predict(uploaded_viz)
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
    st.write("Upload multiple images (normal or defective). These will be saved for retraining, then you can trigger a retrain.")

    category = st.selectbox("MVTec AD category", ["bottle", "cable", "hazelnut", "screw", "metal_nut", "pill", "capsule", "toothbrush", "transistor", "zipper", "carpet", "grid", "leather", "tile", "wood"])

    uploaded_files = st.file_uploader("Choose multiple images", type=["png", "jpg", "jpeg", "bmp", "tif"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected.")
        save_dir = os.path.join(PROJECT_ROOT, "data", "uploads", category, "normal")
        os.makedirs(save_dir, exist_ok=True)
        if st.button("Save uploaded data for retraining"):
            for f in uploaded_files:
                img = Image.open(f).convert("RGB")
                img.save(os.path.join(save_dir, f.name))
            st.success(f"Saved {len(uploaded_files)} images to {save_dir}")

    st.markdown("---")
    st.subheader("Trigger Retraining")
    st.write("Press the button to retrain using the MVTec normal training images plus normal images in `data/uploads/`, then refresh the anomaly threshold.")
    retrain_cat = st.selectbox("Category to retrain on", [category], key="retrain_cat")
    if st.button("🔁 Trigger Retraining", type="primary"):
        with st.spinner("Retraining model... this may take a few minutes."):
            try:
                result = api_retrain(retrain_cat)
                st.success("Retraining complete!")
                st.json(result)
            except Exception as exc:
                st.error(f"Retraining failed: {exc}")

# Page: Model Uptime

elif page == "Model Uptime":
    st.header("Model Uptime & Health")
    st.write("Monitor whether the prediction API is online and responsive.")

    if st.button("Check Health"):
        try:
            health = api_metrics() if False else None
            import requests
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.json(r.json())
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

    st.markdown("---")
    st.subheader("Current Metrics")
    try:
        m = api_metrics()
        st.json(m)
    except Exception as exc:
        st.warning(f"Could not fetch metrics: {exc}")
