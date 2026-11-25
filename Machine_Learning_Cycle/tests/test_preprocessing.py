from PIL import Image

from src import preprocessing


def test_load_uploaded_normal_images_reads_normal_folder(tmp_path, monkeypatch):
    upload_dir = tmp_path / "bottle" / "normal"
    upload_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), color="white").save(upload_dir / "normal.png")
    monkeypatch.setattr(preprocessing, "UPLOAD_DIR", str(tmp_path))

    images = preprocessing.load_uploaded_normal_images("bottle")

    assert len(images) == 1
    assert images[0].size == (4, 4)
