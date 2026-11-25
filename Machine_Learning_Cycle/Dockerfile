# Dockerfile for MVTec AD Anomaly Detection
FROM python:3.11-slim

WORKDIR /app

#System deps (opencv headless, libgl for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==2.9.1+cpu" "torchvision==0.24.1+cpu" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

#Expose API and Streamlit ports
EXPOSE 5000 8501

#Default: run the Flask API. Override command for Streamlit worker if needed.
CMD ["python", "api/app.py"]
