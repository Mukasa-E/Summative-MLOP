# MVTec AD Anomaly Detection

End-to-end MLOps project **manufacturing defect detection** on the [MVTec Anomaly Detection (AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

**Use case:** Detect microscopic scratches, dents, or missing components on industrial parts (screws, bottles, circuit boards, etc.) using anomaly detection — the model learns what "normal" looks like and flags anything that deviates.

Link to the report:https://docs.google.com/document/d/1nrRAbg7c3En7ji9u58jdAX9NjbTyAbPkBZ9zNiNmOpM/edit?tab=t.0

Link to the deployed streamlit ui:https://rodamanomaly.centralindia.cloudapp.azure.com/

Video demo:https://youtu.be/HpXgcdxYPDE?si=aGmFZhQ4od0wAIGK

## Project Structure

Machine_Learning_Cycle/
│
├── README.md                      
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container image for API + UI
├── docker-compose.yml             # Multi-service deployment
├── locustfile.py                  # Load-testing script (flood simulation)
├── retrain.py                     # Retraining trigger script
│
├── notebook/
│   └──summative_mlop.ipynb         # EDA, training, evaluation (≥4 metrics)
│── Report                          #Find report pdf here
├── src/
│   ├── preprocessing.py           # Data acquisition & image preprocessing
│   ├── model.py                   # ConvAutoencoder anomaly model
│   └── prediction.py              # Single-image prediction
│
├── api/
│   └── app.py                     # Flask REST API (/predict, /retrain, /health)
│
├── ui/
│   └── app.py                     # Streamlit web app (predict, visualize, retrain)
│
├── data/
│   ├── anomaly detection data/   # Extracted MVTec AD category folders
│   ├── train/                     # Training images (normal only)
│   ├── test/                      # Test images (normal + anomalous)
│   ├── uploads/                   # Additional normal images uploaded through the UI
│   └── processed/                 # Saved .npy arrays
│
└── models/
    ├── autoencoder_mvtec.pth      # Trained model weights
    └── threshold.npy              # Anomaly decision threshold
```

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Mukasa-E/Summative-MLOP.git
cd Machine_Learning_Cycle
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download MVTec AD

Register and download the dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad, then extract it to `data/anomaly detection data/` so that category folders such as `bottle/train/good` and `bottle/test/` are present. Paths are resolved relative to the project, so commands work when launched from the project root, `api/`, or the notebook directory.

### 3. Preprocess data (one category)

```bash
python -c "
from src.preprocessing import acquire_and_prepare
X_train, X_test, y_test, paths = acquire_and_prepare('bottle', limit=None)
"
```

### 4. Train the model

Or run the complete training, threshold selection, and evaluation pipeline against the real MVTec data:

```bash
python retrain.py --category bottle --epochs 10 --percentile 70
```

By default, retraining uses every available image in the selected category's train and test splits. Training still fits on normal `train/good` images only; anomalous images and the mixed test split are kept out of fitting. To train all 15 categories one after another, run `python retrain.py --category all --epochs 10 --percentile 70`. This can take considerably longer than training one category. Run it at least once for each category before starting inference for that category. The trained bottle checkpoint and threshold are included in `models/` so the grader can run bottle inference without first training; other category checkpoints are created by retraining them. Raw dataset and processed arrays remain excluded from Git.

Each product category is trained and validated separately using its own normal training images, model checkpoint, and threshold. Do not mix unrelated product categories into one model: their normal appearances and anomaly-score distributions differ. Training uses defect-free `train/good` images and any additional verified normal images saved through **Upload & Retrain** under `data/uploads/<category>/normal/`. Do not upload defective images to this directory: all uploaded images are treated as normal training examples. Images submitted on the **Predict** page are inference-only and are not added to training. A normal-only validation split drives early stopping and threshold selection; the category's mixed test images are held back for evaluation. The default 70th-percentile threshold is a recall-oriented operating point and will cause more false alarms than the previous 90th-percentile setting. Confirm the trade-off on validation data and report final test performance without tuning against test labels. The model is tuned to the MVTec image distribution; normal images with different backgrounds, lighting, or viewpoints may still score as anomalous. Each image score is the mean error over the highest-error 0.5% of pixels, which is more sensitive to small localized defects than whole-image average error.

The UI supports all 15 MVTec categories, but each category must be trained before its model can be used for prediction. The upload page requires confirmation that all new training images are normal and limits each image to 10 MB.

```bash
python -c "
import numpy as np
from sklearn.model_selection import train_test_split
from src.model import train_model, select_threshold, predict, anomaly_score

X_train = np.load('data/processed/bottle_X_train.npy')
X_test  = np.load('data/processed/bottle_X_test.npy')
y_test  = np.load('data/processed/bottle_y_test.npy')
X_fit, X_validation = train_test_split(X_train, test_size=0.2, random_state=42)

model, history = train_model(
    X_fit,
    epochs=10,
    validation_data=X_validation,
    category="bottle",
)

# threshold from held-out normal reconstruction errors
scores_normal, _ = anomaly_score(model, X_validation)
thr = select_threshold(scores_normal, percentile=70)

from src.model import save_threshold
save_threshold(thr, category="bottle")

# evaluate
labels, scores = predict(model, X_test, thr)
from sklearn.metrics import classification_report
print(classification_report(y_test, labels))
"
```

### 5. Run the API

```bash
cd api
python app.py
# API available at http://localhost:5000
```

### 6. Run the UI

```bash
streamlit run ui/app.py
# UI available at http://localhost:8501
```

### 7. Run with Docker (recommended)

```bash
docker compose up --build
# API:  http://localhost:5000
# UI:   http://localhost:8501
```

### 8. Publish the UI on an Azure VM

The default Compose setup binds the API and UI to the VM's loopback interface.
To publish the UI, configure a DNS hostname for the VM's public IP, copy
`.env.example` to `.env`, and set `DOMAIN`. Keep `.env` private and do not
commit it. Caddy obtains and renews HTTPS certificates automatically when the
hostname resolves to the VM and inbound TCP ports 80 and 443 are allowed.

The Docker image installs CPU-only PyTorch wheels to avoid downloading CUDA
libraries on CPU-only hosts.


 Start the public profile with:

```bash
docker compose --profile public up --build -d
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check (model uptime) |
| POST | `/predict` | Upload one image → anomaly prediction |
| POST | `/retrain` | Start background retraining of the bottle model |
| GET | `/retrain/status/<job_id>` | Check background retraining status and results |
| GET | `/metrics` | Basic performance metrics |

Example `/predict` (curl):

```bash
curl -X POST -F "file=@data/anomaly detection data/bottle/test/broken_large/000.png" http://localhost:5000/predict
```

Response (image values are PNG data URLs; reconstruction and heatmap are resized to 224×224):

```json
{
  "label": 1,
  "score": 0.0472,
  "threshold": 0.0215,
  "anomaly": true,
  "confidence": 0.89,
  "reconstruction_image": "data:image/png;base64,...",
  "error_heatmap": "data:image/png;base64,...",
  "heatmap_overlay": "data:image/png;base64,..."
}
```

---

## Load Testing with Locust

The Locust script simulates prediction and health requests:

```bash
locust -f locustfile.py --host=http://localhost:5000
```

---

## Notebook

`notebook/summative_mlop.ipynb` contains:

1. **Data acquisition** — loading MVTec AD images
2. **Preprocessing** — resize, normalize, train/test split
3. **Model training** — ConvAutoencoder with early stopping
4. **Evaluation** — accuracy, loss, F1-score, precision, recall, ROC-AUC
5. **Visualizations** — reconstructions, error heatmaps, feature insights (≥3 features)
6. **Prediction demo** — single-image inference

The notebook is executed with saved loss curves, sample images, reconstruction/error visualizations, and evaluation metrics.

---
