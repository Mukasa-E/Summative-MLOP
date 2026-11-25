"""
Retraining script — triggered by UI / API / scheduled job.
Re-acquires data, retrains the autoencoder on normal images, selects a new threshold,
and saves model + threshold to models/.
"""

import os
import sys
import argparse
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import acquire_and_prepare
from src.model import train_model, anomaly_score, select_threshold
from src.prediction import save_threshold


def retrain(category="bottle", limit=300, epochs=10, percentile=90):
    print(f"[RETRAIN] Category: {category} | limit: {limit} | epochs: {epochs}")

    # 1. Acquire + preprocess the selected MVTec category
    X_train, X_test, y_test, _ = acquire_and_prepare(category, limit=limit)
    print(f"[RETRAIN] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    X_fit, X_validation = train_test_split(
        X_train,
        test_size=0.2,
        random_state=42,
    )

    # 2. Train autoencoder on normal images only
    model, _ = train_model(
        X_fit,
        epochs=epochs,
        validation_data=X_validation,
    )

    # 3. Select threshold from held-out normal reconstruction errors
    scores_normal, _ = anomaly_score(model, X_validation)
    thr = select_threshold(scores_normal, percentile=percentile)
    save_threshold(thr)

    # 4. Evaluate on test set
    from src.model import predict
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    labels, scores = predict(model, X_test, thr)
    metrics = {
        "accuracy": float(accuracy_score(y_test, labels)),
        "precision": float(precision_score(y_test, labels, zero_division=0)),
        "recall": float(recall_score(y_test, labels, zero_division=0)),
        "f1": float(f1_score(y_test, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
    }
    print(f"[RETRAIN] Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--percentile", type=int, default=90)
    args = parser.parse_args()
    retrain(args.category, args.limit, args.epochs, args.percentile)
