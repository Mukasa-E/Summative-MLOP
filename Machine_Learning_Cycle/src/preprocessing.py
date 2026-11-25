"""
Data Acquisition & Preprocessing for MVTec AD Anomaly Detection.
---------------------------------------------------------------
Loads raw MVTec AD images, resizes/normalizes them, and splits into
train (normal-only) and test sets. Supports saving/loading processed
arrays so the pipeline can be re-run without re-reading 5GB of images.
"""

import os
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
NORMAL_CLASS = "good"                 # MVTec AD folder name for defect-free images
UPLOAD_NORMAL_CLASS = "normal"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "anomaly detection data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")


def ensure_dirs():
    """Create output directories if missing."""
    for d in [PROCESSED_DIR, TRAIN_DIR, TEST_DIR, os.path.join(PROJECT_ROOT, "models")]:
        os.makedirs(d, exist_ok=True)


def load_mvtec_images(category, split="train", limit=None):
    """
    Load images from a single MVTec AD category folder.

    Parameters
    ----------
    category : str
        MVTec AD category name, e.g. "bottle", "hazelnut", "screw".
    split : str
        "train" or "test".
    limit : int or None
        Max number of images to load (useful for quick experiments).

    Returns
    -------
    images : list of PIL.Image
    labels : list of int   (0 = normal/good, 1 = anomalous/defective)
    paths  : list of str
    """
    base = os.path.join(RAW_DATA_ROOT, category, split)
    images, labels, paths = [], [], []

    if not os.path.isdir(base):
        raise FileNotFoundError(f"MVTec {split} folder not found: {base}")

    # Walk all subfolders (good / defective types)
    for root, directories, files in os.walk(base):
        directories.sort()
        for fname in sorted(files):
            if limit is not None and len(images) >= limit:
                break
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                images.append(img)
                paths.append(fpath)
                # Label: 0 if the leaf folder is 'good', else 1 (anomalous)
                leaf = os.path.basename(root).lower()
                labels.append(0 if leaf == NORMAL_CLASS else 1)
            except Exception as exc:
                print(f"[SKIP] {fpath}: {exc}")

    if not images:
        raise ValueError(f"No supported images found in {base}")

    return images, labels, paths


def load_uploaded_normal_images(category):
    """Load additional defect-free images uploaded for one category."""
    directory = os.path.join(UPLOAD_DIR, category, UPLOAD_NORMAL_CLASS)
    if not os.path.isdir(directory):
        return []

    images = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        path = os.path.join(directory, fname)
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        except Exception as exc:
            print(f"[SKIP] {path}: {exc}")
    return images


def preprocess_image(img, size=IMG_SIZE):
    """Resize + normalize a single PIL image to a float32 numpy array (HWC, 0-1)."""
    img = img.resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def preprocess_batch(images):
    """Preprocess a list of PIL images into a single numpy array (N, H, W, C)."""
    if not images:
        raise ValueError("Cannot preprocess an empty image list")
    return np.stack([preprocess_image(im) for im in images], axis=0)


def acquire_and_prepare(category, test_size=0.2, random_state=42, limit=None):
    """
    Full acquisition + preprocessing pipeline for one MVTec AD category.

    Train set contains ONLY normal (good) images — this is unsupervised
    anomaly detection: the model learns what "normal" looks like.
    Test set contains both normal and anomalous images for evaluation.
    """
    ensure_dirs()

    print(f"[INFO] Loading train images for '{category}'...")
    train_imgs, train_lbls, _ = load_mvtec_images(category, "train", limit=limit)

    print(f"[INFO] Loading test images for '{category}'...")
    test_imgs, test_lbls, test_paths = load_mvtec_images(category, "test", limit=limit)

    # Keep only normal images for training (unsupervised AD)
    train_normal = [im for im, lbl in zip(train_imgs, train_lbls) if lbl == 0]
    uploaded_normal = load_uploaded_normal_images(category)
    train_normal.extend(uploaded_normal)
    if not train_normal:
        raise ValueError(f"No normal training images found for category '{category}'")
    print(f"[INFO] Normal train images: {len(train_normal)} "
          f"(including {len(uploaded_normal)} uploaded) | "
          f"Test images: {len(test_imgs)} (normal={sum(1 for l in test_lbls if l == 0)}, "
          f"anomalous={sum(1 for l in test_lbls if l == 1)})")

    X_train = preprocess_batch(train_normal)
    X_test = preprocess_batch(test_imgs)
    y_test = np.array(test_lbls, dtype=np.int64)

    # Save processed arrays
    np.save(os.path.join(PROCESSED_DIR, f"{category}_X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, f"{category}_X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, f"{category}_y_test.npy"), y_test)
    np.save(os.path.join(PROCESSED_DIR, f"{category}_test_paths.npy"), np.array(test_paths, dtype=object))

    # Also copy raw files into data/train and data/test for the repo structure
    _copy_images(train_normal, os.path.join(TRAIN_DIR, category, "normal"))
    _copy_images(test_imgs, os.path.join(TEST_DIR, category))

    print(f"[INFO] Saved processed data to {PROCESSED_DIR}")
    return X_train, X_test, y_test, test_paths


def _copy_images(images, dest_dir):
    """Helper: save PIL images to disk (preserves repo structure expected by PDF)."""
    os.makedirs(dest_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(dest_dir, f"img_{i:05d}.png"))


def load_processed(category):
    """Load previously saved processed arrays."""
    X_train = np.load(os.path.join(PROCESSED_DIR, f"{category}_X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, f"{category}_X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, f"{category}_y_test.npy"))
    test_paths = np.load(os.path.join(PROCESSED_DIR, f"{category}_test_paths.npy"), allow_pickle=True)
    return X_train, X_test, y_test, test_paths


if __name__ == "__main__":
    # Example usage (uncomment and point RAW_DATA_ROOT to the extracted MVTec AD):
    # X_train, X_test, y_test, paths = acquire_and_prepare("bottle", limit=200)
    pass
