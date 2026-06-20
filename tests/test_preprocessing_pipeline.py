import os
import sys
import tempfile
import json
import pickle
import pytest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts.convert_datasets import ConvertedScamRecord, ConversationTurn
from src.data_pipeline.preprocessing_pipeline import (
    validate_record,
    join_turns,
    preprocess_text,
    preprocess_pipeline
)

@pytest.fixture
def sample_record_dict():
    return {
        "id": "12345678-1234-1234-1234-1234567890ab",
        "language": "en",
        "source": "test_source",
        "conversation": [
            {"speaker": "Scammer", "text": "Please transfer Rs. 5000 to my account SBI."},
            {"speaker": "Victim", "text": "Why? Send me your UPI or IFSC."}
        ],
        "label": "Bank KYC Scam",
        "risk_score": 85.5,
        "tags": {
            "authority_impersonation": False,
            "urgency": True,
            "fear": False,
            "money_request": True,
            "otp_request": False,
            "bank_details_request": True,
            "qr_request": False,
            "keep_secret": False,
            "remote_access_request": False
        }
    }

def test_validate_record_valid(sample_record_dict):
    record = validate_record(sample_record_dict)
    assert isinstance(record, ConvertedScamRecord)
    assert record.id == "12345678-1234-1234-1234-1234567890ab"
    assert record.risk_score == 85.5
    assert len(record.conversation) == 2

def test_validate_record_invalid(sample_record_dict):
    invalid_dict = sample_record_dict.copy()
    invalid_dict["risk_score"] = 150.0  # Invalid as risk_score must be <= 100.0
    with pytest.raises(Exception):
        validate_record(invalid_dict)

def test_join_turns(sample_record_dict):
    record = validate_record(sample_record_dict)
    joined = join_turns(record)
    expected = "Scammer: Please transfer Rs. 5000 to my account SBI. Victim: Why? Send me your UPI or IFSC."
    assert joined == expected

def test_preprocess_text_lowercasing_and_basic_cleaning():
    text = "Hello WORLD! This is a TEST."
    cleaned = preprocess_text(text)
    assert cleaned == "hello world this is a test"

def test_preprocess_text_preserves_otp():
    text = "Your OTP is 482910. Do not share."
    cleaned = preprocess_text(text)
    assert "482910" in cleaned
    assert "otp" in cleaned

def test_preprocess_text_preserves_aadhaar():
    text = "Provide Aadhaar number 1234-5678-9012 for validation."
    cleaned = preprocess_text(text)
    assert "1234-5678-9012" in cleaned
    assert "aadhaar" in cleaned

def test_preprocess_text_preserves_pan():
    text = "My PAN number is abcde1234f."
    cleaned = preprocess_text(text)
    assert "abcde1234f" in cleaned

def test_preprocess_text_preserves_upi():
    text = "Send money to fraudster@okaxis or upi ID."
    cleaned = preprocess_text(text)
    assert "fraudster@okaxis" in cleaned
    assert "upi" in cleaned

def test_preprocess_text_preserves_ifsc():
    text = "Use IFSC code SBIN0012345."
    cleaned = preprocess_text(text)
    # The original was SBIN0012345, which will be lowercased to sbin0012345
    assert "sbin0012345" in cleaned

def test_preprocess_text_preserves_bank_names():
    text = "Is it State Bank of India or HDFC or Axis Bank?"
    cleaned = preprocess_text(text)
    assert "state bank of india" in cleaned
    assert "hdfc" in cleaned
    assert "axis bank" in cleaned

def test_preprocess_text_preserves_currency_symbols():
    text = "Please pay ₹5000 or Rs. 1000 or $50."
    cleaned = preprocess_text(text)
    assert "₹5000" in cleaned or "₹" in cleaned
    assert "rs. 1000" in cleaned or "rs." in cleaned
    assert "$50" in cleaned or "$" in cleaned

def test_preprocess_text_preserves_phone_numbers():
    text = "Call me on +91-98765-43210 or 09876543210."
    cleaned = preprocess_text(text)
    assert "+91-98765-43210" in cleaned
    assert "09876543210" in cleaned

def test_preprocess_text_punctuation_removal_and_whitespace():
    # Verify unnecessary punctuation like commas, periods, question marks are removed,
    # but phone separators/currency symbols are kept.
    text = "Hello, Scammer!!! Pay $50 now? Yes, please."
    cleaned = preprocess_text(text)
    assert cleaned == "hello scammer pay $50 now yes please"

def test_preprocess_pipeline(sample_record_dict):
    # Create temp files for input/output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "train_dataset.json"
        output_file = Path(temp_dir) / "preprocessed_dataset.pkl"

        # Create multiple samples so that stratification works (need at least 2 for split)
        records = []
        for i in range(10):
            rec = sample_record_dict.copy()
            rec["id"] = f"12345678-1234-1234-1234-1234567890a{i}"
            # Give half of them a different label so we have stratification check
            if i % 2 == 0:
                rec["label"] = "OTP Scam"
            records.append(rec)

        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(records, f)

        # Run pipeline with split
        res = preprocess_pipeline(
            input_path=str(input_file),
            output_path=str(output_file),
            test_size=0.2,
            random_state=42,
            split_dataset=True
        )

        assert output_file.exists()
        
        # Load output
        with open(output_file, "rb") as f:
            data = pickle.load(f)

        assert "train" in data
        assert "test" in data
        assert len(data["train"]["texts"]) == 8
        assert len(data["test"]["texts"]) == 2
        
        # Verify fields in train
        assert "texts" in data["train"]
        assert "labels" in data["train"]
        assert "risk_scores" in data["train"]
        assert "behavior_tags" in data["train"]

        # Verify that preprocessing cleaned the texts
        for text in data["train"]["texts"]:
            assert "scammer please transfer rs. 5000 to my account sbi victim why send me your upi or ifsc" in text
