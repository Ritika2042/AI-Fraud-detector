import sys
from pathlib import Path
import pytest

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.reasoning.evidence_extractor import EvidenceExtractor
from src.reasoning.risk_engine import RiskEngine
from src.reasoning.recommendation_engine import RecommendationEngine
from src.reasoning.confidence_calibrator import ConfidenceCalibrator
from src.inference import ScamPredictor

def test_evidence_extractor_categories():
    extractor = EvidenceExtractor()
    
    # Test cases mapping input texts to expected detected indicator categories
    test_cases = [
        ("Please share the OTP sent to your phone.", "otp_request"),
        ("Enter your UPI PIN to receive the money.", "upi_pin_request"),
        ("Scan this QR code to claim your cashback voucher.", "qr_code_request"),
        ("Your bank KYC has expired. Complete verification now.", "bank_kyc"),
        ("Send me your Aadhaar card or PAN details.", "aadhaar_pan_mentions"),
        ("This is CBI calling from Delhi headquarters.", "police_cbi_impersonation"),
        ("You are under digital arrest. Do not hang up the call.", "digital_arrest"),
        ("Transfer funds to the agent's account immediately.", "money_transfer_request"),
        ("Send USDT or Bitcoin to my crypto wallet.", "cryptocurrency_mentions"),
        ("You must act immediately and pay right now.", "urgency_phrases"),
        ("If you do not cooperate you will face jail and arrest warrant.", "threat_language"),
        ("Pay customs duty for your foreign gift parcel.", "gift_customs_request"),
        ("Pay the documentation processing fee for loan approval.", "loan_processing_fee"),
        ("We offer guaranteed returns on your stock investment.", "investment_promises")
    ]
    
    for text, category in test_cases:
        res = extractor.extract_evidence(text)
        assert res["evidence"][category]["detected"] is True, f"Failed to detect {category} in text: '{text}'"
        assert res["evidence"][category]["snippet"] in text

def test_risk_engine_calculation():
    risk_engine = RiskEngine()
    
    # Mock evidence input
    mock_evidence = {
        "upi_pin_request": {"detected": True},        # Weight 20
        "digital_arrest": {"detected": True},         # Weight 20
        "otp_request": {"detected": False},
        "qr_code_request": {"detected": True},        # Weight 15
        "police_cbi_impersonation": {"detected": False},
        "bank_kyc": {"detected": True},               # Weight 10
        "money_transfer_request": {"detected": False},
        "threat_language": {"detected": False},
        "gift_customs_request": {"detected": False},
        "loan_processing_fee": {"detected": False},
        "investment_promises": {"detected": False},
        "aadhaar_pan_mentions": {"detected": True},    # Weight 5
        "cryptocurrency_mentions": {"detected": False},
        "urgency_phrases": {"detected": True}          # Weight 5
    }
    
    # Expected score = 20 + 20 + 15 + 10 + 5 + 5 = 75.0
    res = risk_engine.compute_risk(mock_evidence)
    assert res["risk_score"] == 75.0
    assert res["risk_level"] == "High"

def test_risk_engine_capping():
    risk_engine = RiskEngine()
    
    # Mock evidence that exceeds 100
    mock_evidence = {cat: {"detected": True} for cat in risk_engine.weights.keys()}
    res = risk_engine.compute_risk(mock_evidence)
    assert res["risk_score"] == 100.0
    assert res["risk_level"] == "Critical"

def test_risk_engine_low_risk():
    risk_engine = RiskEngine()
    
    mock_evidence = {
        "aadhaar_pan_mentions": {"detected": True},    # Weight 5
        "urgency_phrases": {"detected": True}          # Weight 5
    }
    res = risk_engine.compute_risk(mock_evidence)
    assert res["risk_score"] == 10.0
    assert res["risk_level"] == "Low"

def test_recommendation_engine():
    rec_engine = RecommendationEngine()
    
    rec_otp = rec_engine.get_recommendation("OTP Scam")
    assert "Never share OTPs" in rec_otp
    
    rec_digital = rec_engine.get_recommendation("Digital Arrest Scam")
    assert "digital arrest" in rec_digital.lower()
    
    rec_safe = rec_engine.get_recommendation("Safe")
    assert "No scam detected" in rec_safe
    
    rec_default = rec_engine.get_recommendation("Unknown Category")
    assert "credentials" in rec_default

def test_confidence_calibrator():
    calibrator = ConfidenceCalibrator()
    
    assert calibrator.calibrate(0.55) == "Low"
    assert calibrator.calibrate(0.70) == "Medium"
    assert calibrator.calibrate(0.95) == "High"

def test_inference_predictor_integration():
    # If the baseline SVM model is trained, let's load it and run a quick prediction.
    # Otherwise, this test is skipped or runs against whatever model is loaded.
    predictor = ScamPredictor()
    if not predictor.is_model_loaded():
        pytest.skip("Scam predictor model not loaded. Skipping integration test.")
        
    text = "CBI officer calling. You are under digital arrest for money laundering. Pay rupees 50000 immediately."
    res = predictor.predict_single(text)
    
    assert "error" not in res
    assert "prediction" in res
    assert "confidence" in res
    assert "confidence_level" in res
    assert "risk_score" in res
    assert "risk_level" in res
    assert "evidence" in res
    assert "top_keywords" in res
    assert "recommendation" in res
    
    # Confirm values of new fields are typed correctly
    assert isinstance(res["confidence_level"], str)
    assert isinstance(res["risk_score"], float)
    assert isinstance(res["risk_level"], str)
    assert isinstance(res["evidence"], dict)
    assert isinstance(res["top_keywords"], list)
    assert isinstance(res["recommendation"], str)
    
    # Specific checks on digital arrest indicator
    assert res["evidence"]["digital_arrest"]["detected"] is True
    assert "digital arrest" in res["evidence"]["digital_arrest"]["snippet"]
