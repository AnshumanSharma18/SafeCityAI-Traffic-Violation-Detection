"""SafeCityAI - API Unit Tests.

Tests API endpoints without requiring actual trained model weights.
"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.server import app, WEIGHTS_PATH

client = TestClient(app)


def test_health_endpoint():
    """Test GET /health returns 200 OK and valid health response schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_exists" in data
    assert "model_loaded" in data
    assert data["supported_classes"] == ["Helmet", "LicensePlate", "NoHelmet"]


def test_predict_missing_model_behavior():
    """Test POST /predict returns HTTP 503 when model weights file best.pt is absent."""
    # Ensure test simulates missing best.pt
    with patch.object(Path, "is_file", return_value=False):
        # Create dummy image buffer
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "best.pt" in data["detail"]


def test_predict_invalid_file_type():
    """Test POST /predict returns HTTP 400 when an invalid non-image file is uploaded."""
    text_content = b"This is a text file, not an image."
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", io.BytesIO(text_content), "text/plain")}
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Invalid file type" in data["detail"]


def test_predict_corrupted_image_file():
    """Test POST /predict returns HTTP 400 when image content is corrupted."""
    corrupted_bytes = b"GIF89a corrupted image header data"
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", io.BytesIO(corrupted_bytes), "image/png")}
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
