"""
Locust load-testing script for the MVTec AD prediction API.

Simulates a flood of requests to measure latency and response time.
Run:
    locust -f locustfile.py --host=http://localhost:5000
Then open http://localhost:8089 and configure users / spawn rate.

Test with different Docker container counts:
    docker-compose up --scale api=1   # baseline
    docker-compose up --scale api=2
    docker-compose up --scale api=4
Record average response time, RPS, and failure rate for each.
"""

import os
from locust import HttpUser, task, between
import random


class MlPredictUser(HttpUser):
    """Simulated users hitting the /predict endpoint with sample images."""

    wait_time = between(0.5, 2.0)

    # A few sample image paths from data/test (edit to match your extracted dataset)
    SAMPLE_IMAGES = [
        "data/test/bottle/img_00000.png",
        "data/test/bottle/img_00001.png",
        "data/test/bottle/img_00002.png",
    ]

    @task(3)
    def predict_normal(self):
        """Predict on a random sample image."""
        img_path = random.choice(self.SAMPLE_IMAGES)
        if not os.path.exists(img_path):
            return
        with open(img_path, "rb") as f:
            self.client.post("/predict", files={"file": (img_path, f, "image/png")}, name="/predict")

    @task(1)
    def health_check(self):
        """Occasional health check (models uptime)."""
        self.client.get("/health", name="/health")



