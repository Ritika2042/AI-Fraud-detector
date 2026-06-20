import os
import tempfile
import yaml
import pytest
from src.config import load_config
from src.modeling.model import TfidfScamClassifier
from src.inference import ScamPredictor

@pytest.fixture(scope="module")
def dummy_model_and_config():
    """Module-level fixture to train a dummy model and dump a temporary config."""
    # 1. Create a tiny dataset to train a dummy model
    texts = [
        "Please update your bank KYC details now or account will be suspended.",
        "Hey buddy, are we meeting for coffee today at 5 PM?",
        "Send me the OTP verification code sent to your phone immediately.",
        "Your Amazon delivery package has been dispatched and is on the way.",
    ]
    labels = ["Bank KYC Scam", "Safe", "OTP Scam", "Safe"]
    
    model = TfidfScamClassifier(max_features=100)
    model.fit(texts, labels)
    
    # 2. Save dummy model to a temporary folder
    temp_dir = tempfile.TemporaryDirectory()
    temp_model_path = os.path.join(temp_dir.name, "test_model.pkl")
    model.save(temp_model_path)
    
    # 3. Create a temporary YAML config file pointing to the temporary model path
    orig_config = load_config()
    
    config_dict = {
        "project_name": "test_scam_detector",
        "data": {
            "raw_data_path": orig_config.data.raw_data_path,
            "processed_data_path": orig_config.data.processed_data_path,
            "test_size": 0.25,
            "random_state": 42
        },
        "model": {
            "model_path": temp_model_path,
            "metrics_path": os.path.join(temp_dir.name, "metrics.json"),
            "model_type": "tfidf_logistic",
            "tfidf": {
                "max_features": 100,
                "ngram_range": [1, 2],
                "lowercase": True
            },
            "classifier": {
                "C": 1.0,
                "max_iter": 100,
                "class_weight": "balanced",
                "random_state": 42
            }
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8999,
            "debug": False,
            "cors_origins": ["*"]
        },
        "logging": {
            "log_level": "DEBUG",
            "log_file": os.path.join(temp_dir.name, "test.log"),
            "backup_count": 1,
            "max_bytes": 100000
        }
    }
    
    temp_config_path = os.path.join(temp_dir.name, "config.yaml")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
        
    yield temp_config_path, temp_model_path, temp_dir
    
    # Clean up temp files
    temp_dir.cleanup()

def test_singleton_pattern(dummy_model_and_config):
    config_path, _, _ = dummy_model_and_config
    
    # Clear the existing singleton instance to enforce reload with test config
    ScamPredictor._instance = None
    
    predictor1 = ScamPredictor(config_path=config_path)
    predictor2 = ScamPredictor(config_path=config_path)
    
    assert predictor1 is predictor2

def test_inference_and_explanation(dummy_model_and_config):
    config_path, _, _ = dummy_model_and_config
    
    ScamPredictor._instance = None
    predictor = ScamPredictor(config_path=config_path)
    
    # Verify model is successfully loaded
    assert predictor.is_model_loaded() is True
    
    # Predict on a KYC scam sample text
    response = predictor.predict_single("Update my bank KYC details urgently.")
    
    assert "error" not in response
    assert response["prediction"] == "Bank KYC Scam"
    assert response["confidence"] > 0.0
    assert "Bank KYC Scam" in response["probabilities"]
    
    # Verify explanation triggers have extracted keywords
    assert len(response["explanations"]) > 0
    words = [item["word"] for item in response["explanations"]]
    # The TF-IDF + Logistic regression coefficients should pick up key trigger words
    assert any(w in words for w in ["kyc", "bank", "update"])

def test_batch_inference(dummy_model_and_config):
    config_path, _, _ = dummy_model_and_config
    
    ScamPredictor._instance = None
    predictor = ScamPredictor(config_path=config_path)
    
    texts = [
        "Please send the security verification OTP code.",
        "Hey there, are we going to play football tonight?"
    ]
    
    results = predictor.predict_batch(texts)
    
    assert len(results) == 2
    assert results[0]["prediction"] == "OTP Scam"
    assert results[1]["prediction"] == "Safe"
    assert "error" not in results[0]
    assert "error" not in results[1]
