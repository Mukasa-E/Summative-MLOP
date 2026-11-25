"""
Anomaly Detection Model: Convolutional Autoencoder.
---------------------------------------------------
Trains on normal (defect-free) images only. At inference time, the
reconstruction error (MSE between input and output) is used as the
anomaly signal: higher pixel error  =>  more likely to be anomalous.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_mvtec.pth")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvAutoencoder(nn.Module):
    """Simple convolutional autoencoder for 224x224 RGB images."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112x112
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),    # 224x224
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_model(
    X_train,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    verbose=True,
    validation_data=None,
    patience=3,
):
    """
    Train the autoencoder on normal images only, optionally using validation
    data for early stopping.
    Returns the trained model and a list of epoch losses.
    """
    if len(X_train) == 0:
        raise ValueError("X_train must contain at least one image")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if validation_data is not None and patience < 1:
        raise ValueError("patience must be at least 1")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(42)
    training_sample_count = len(X_train)

    # (N, H, W, C) -> (N, C, H, W)
    X = np.transpose(X_train, (0, 3, 1, 2))
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation_loader = None
    validation_sample_count = 0
    if validation_data is not None:
        if len(validation_data) == 0:
            raise ValueError("validation_data must contain at least one image")
        validation_sample_count = len(validation_data)
        X_validation = np.transpose(validation_data, (0, 3, 1, 2))
        validation_dataset = TensorDataset(
            torch.from_numpy(X_validation),
            torch.from_numpy(X_validation),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / training_sample_count
        history.append(avg_loss)
        if verbose:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        if validation_loader is not None:
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch_x, _ in validation_loader:
                    batch_x = batch_x.to(DEVICE)
                    validation_loss += criterion(model(batch_x), batch_x).item() * batch_x.size(0)
            validation_loss /= validation_sample_count
            if verbose:
                print(f"[Epoch {epoch+1}/{epochs}] Validation loss: {validation_loss:.6f}")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"[INFO] Early stopping after epoch {epoch+1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")
    return model, history


def load_model():
    """Load a trained autoencoder from disk."""
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def anomaly_score(model, X):
    """
    Compute per-sample anomaly scores as the mean of the top 0.5% pixel errors.
    Returns scores (N,) and reconstructions (N, C, H, W).
    """
    model.eval()
    X_t = np.transpose(X, (0, 3, 1, 2))
    with torch.no_grad():
        inp = torch.from_numpy(X_t).to(DEVICE)
        recon = model(inp)
        # Defects are often localized, so score using the highest-error 0.5% of pixels.
        pixel_errors = ((recon - inp) ** 2).mean(dim=1).flatten(start_dim=1)
        top_pixels = max(1, int(np.ceil(pixel_errors.shape[1] * 0.005)))
        errors = pixel_errors.topk(top_pixels, dim=1).values.mean(dim=1).cpu().numpy()
    return errors, recon.cpu().numpy()


def select_threshold(scores_normal, percentile=90):
    """
    Choose anomaly threshold from normal validation reconstruction errors.
    Default: 90th percentile of held-out normal errors.
    """
    threshold = np.percentile(scores_normal, percentile)
    print(f"[INFO] Threshold (p{percentile}) = {threshold:.6f}")
    return threshold


def save_threshold(value, path=THRESHOLD_PATH):
    """Persist the anomaly threshold to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(value, dtype=np.float64))
    print(f"[INFO] Threshold saved to {path}")


def load_threshold(path=THRESHOLD_PATH):
    """Load the persisted anomaly threshold."""
    values = np.asarray(np.load(path)).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one threshold value in {path}, found {values.size}")
    return float(values[0])


def predict(model, X, threshold):
    """
    Predict anomaly labels for a batch.
    Returns: (labels, scores) where 1 = anomalous, 0 = normal.
    """
    scores, _ = anomaly_score(model, X)
    labels = (scores > threshold).astype(np.int64)
    return labels, scores


if __name__ == "__main__":
    pass
