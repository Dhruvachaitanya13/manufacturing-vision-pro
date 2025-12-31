import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api import app

# Initialize Test Client
client = TestClient(app)

def test_root_endpoint():
    """Test if the API root returns the correct service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Manufacturing Vision Pro"
    assert "version" in data

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert isinstance(data["model_loaded"], bool)

@patch("src.api.engine")  # Mock the global engine instance in api.py
def test_inspect_endpoint_success(mock_engine):
    """Test a successful image inspection."""
    
    # 1. Setup Mock Response
    mock_response = {
        "filename": "test_image.jpg",
        "status": "REJECT",
        "inference_time_ms": 15.5,
        "count": 1,
        "detections": [
            {
                "label": "scratch",
                "confidence": 0.95,
                "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50}
            }
        ]
    }
    mock_engine.predict.return_value = mock_response

    # 2. Create Dummy Image File
    file_content = b"fake_image_bytes"
    files = {"file": ("test_image.jpg", file_content, "image/jpeg")}

    # 3. Call API
    response = client.post("/inspect", files=files)

    # 4. Assertions
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "REJECT"
    assert json_resp["count"] == 1
    assert json_resp["detections"][0]["label"] == "scratch"

def test_inspect_invalid_file_type():
    """Test that the API rejects non-image files."""
    files = {"file": ("test.txt", b"text content", "text/plain")}
    response = client.post("/inspect", files=files)
    assert response.status_code == 415
    assert "Only JPEG/PNG" in response.json()["detail"]