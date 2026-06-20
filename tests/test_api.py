import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.inference import ScamPredictor
from tests.test_inference import dummy_model_and_config

@pytest.fixture
def client(dummy_model_and_config):
    """Fixture that initializes the test client and configures the singleton predictor."""
    config_path, _, _ = dummy_model_and_config
    
    # Force initialize the singleton with the test config
    ScamPredictor._instance = None
    ScamPredictor(config_path=config_path)
    
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["project"] == "test_scam_detector"

def test_predict_endpoint_success(client):
    payload = {
        "conversation": "Your bank account is blocked. You must verify your account immediately by sharing your card number."
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Bank KYC Scam"
    assert data["confidence"] > 0.0
    assert "risk" in data
    assert data["risk"] in ["Critical", "High", "Medium", "Low"]
    assert "reasons" in data
    assert len(data["reasons"]) > 0
    # The KYC indicator triggers should match bank or kyc keywords
    assert any("bank" in r.lower() or "kyc" in r.lower() for r in data["reasons"])

def test_predict_endpoint_validation_error(client):
    # Too short text (less than 5 characters) should fail validation
    payload = {"conversation": "K"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error

def test_predict_batch_endpoint(client):
    payload = {
        "conversations": [
            "Urgently update bank KYC details.",
            "Are we meeting for lunch at 12 PM?"
        ]
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["prediction"] == "Bank KYC Scam"
    assert data["results"][0]["risk"] in ["Critical", "High", "Medium", "Low"]
    assert data["results"][1]["prediction"] == "Safe"

def test_train_endpoint(client):
    response = client.post("/api/v1/train")
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert "scheduled" in data["message"]
