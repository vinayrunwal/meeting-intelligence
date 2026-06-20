from io import BytesIO
import sys

import pytest

@pytest.fixture
def app():
    from src.api.app import create_app

    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    yield app

@pytest.fixture
def client(app):
    return app.test_client()

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data

def test_list_jobs(client):
    """Test listing jobs endpoint."""
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "jobs" in data["data"]

def test_upload_no_file(client):
    """Test uploading without a file part."""
    response = client.post("/api/v1/upload", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "error" in data


def test_upload_reports_missing_ml_dependencies(client, monkeypatch):
    """A lightweight environment should fail uploads clearly before saving jobs."""
    monkeypatch.setitem(sys.modules, "src.pipeline.orchestrator", None)
    response = client.post(
        "/api/v1/upload",
        data={"file": (BytesIO(b"not-real-audio"), "meeting.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    data = response.get_json()
    assert data["success"] is False
    assert "dependencies are not installed" in data["error"]
