import json
import pytest
from pathlib import Path
from scripts.convert_datasets import ConvertedScamRecord, ConversationTurn
from scripts.merge_datasets import get_conversation_key, merge_and_deduplicate, compute_statistics

def test_get_conversation_key():
    turns1 = [
        ConversationTurn(speaker="Scammer", text="Hello there! "),
        ConversationTurn(speaker="Victim", text=" Yes? ")
    ]
    turns2 = [
        ConversationTurn(speaker="Scammer", text="hello there!"),
        ConversationTurn(speaker="Victim", text="yes?")
    ]
    turns3 = [
        ConversationTurn(speaker="Scammer", text="hello there!"),
        ConversationTurn(speaker="Victim", text="no?")
    ]
    
    key1 = get_conversation_key(turns1)
    key2 = get_conversation_key(turns2)
    key3 = get_conversation_key(turns3)
    
    assert key1 == key2  # Case insensitive, whitespace stripped
    assert key1 != key3  # Different text

def test_merge_and_deduplicate(tmp_path):
    # Setup temporary directories for processed and synthetic files
    processed_dir = tmp_path / "processed"
    synthetic_dir = tmp_path / "synthetic"
    processed_dir.mkdir()
    synthetic_dir.mkdir()
    
    # 1. Write unique records
    rec1 = {
        "id": "uuid-1111",
        "language": "en",
        "source": "source_a",
        "conversation": [{"speaker": "Scammer", "text": "hello kyc expired"}],
        "label": "Bank KYC Scam",
        "risk_score": 80.0,
        "tags": {"authority_impersonation": True}
    }
    
    rec2 = {
        "id": "uuid-2222",
        "language": "en",
        "source": "source_b",
        "conversation": [{"speaker": "Scammer", "text": "digital arrest threat"}],
        "label": "Digital Arrest Scam",
        "risk_score": 98.0,
        "tags": {"authority_impersonation": True, "fear": True}
    }
    
    # 2. Duplicate ID record (same ID as rec1, different conversation)
    rec_dup_id = {
        "id": "uuid-1111",
        "language": "en",
        "source": "source_a",
        "conversation": [{"speaker": "Scammer", "text": "completely different conversation"}],
        "label": "Safe",
        "risk_score": 0.0,
        "tags": {}
    }
    
    # 3. Duplicate conversation record (different ID, same conversation as rec2)
    rec_dup_conv = {
        "id": "uuid-3333",
        "language": "en",
        "source": "source_c",
        "conversation": [{"speaker": "Scammer", "text": "digital arrest threat"}],
        "label": "Digital Arrest Scam",
        "risk_score": 98.0,
        "tags": {"authority_impersonation": True, "fear": True}
    }
    
    # Write files
    with open(processed_dir / "real.json", "w", encoding="utf-8") as f:
        json.dump([rec1, rec2], f)
        
    with open(synthetic_dir / "fake.json", "w", encoding="utf-8") as f:
        json.dump([rec_dup_id, rec_dup_conv], f)
        
    # Run merge
    merged = merge_and_deduplicate(processed_dir, synthetic_dir)
    
    # Total unique records should be exactly 2 (rec1 and rec2)
    assert len(merged) == 2
    assert {r.id for r in merged} == {"uuid-1111", "uuid-2222"}

def test_compute_statistics():
    rec1 = ConvertedScamRecord(
        id="1",
        language="en",
        source="src_a",
        conversation=[
            ConversationTurn(speaker="Scammer", text="hello OTP please"),
            ConversationTurn(speaker="Victim", text="the code is 1234")
        ],
        label="OTP Scam",
        risk_score=85.0,
        tags={"otp_request": True}
    )
    
    rec2 = ConvertedScamRecord(
        id="2",
        language="en",
        source="src_b",
        conversation=[
            ConversationTurn(speaker="Scammer", text="we are CBI police officers")
        ],
        label="Digital Arrest Scam",
        risk_score=95.0,
        tags={"authority_impersonation": True}
    )
    
    rec3 = ConvertedScamRecord(
        id="3",
        language="en",
        source="src_a",
        conversation=[
            ConversationTurn(speaker="Scammer", text="kyc expire update details"),
            ConversationTurn(speaker="Victim", text="ok doing it"),
            ConversationTurn(speaker="Scammer", text="thank you")
        ],
        label="OTP Scam",  # 2 OTP Scams, 1 Digital Arrest
        risk_score=60.0,
        tags={"bank_details_request": True}
    )
    
    stats = compute_statistics([rec1, rec2, rec3])
    
    assert stats["total_records"] == 3
    assert stats["label_counts"]["OTP Scam"] == 2
    assert stats["label_counts"]["Digital Arrest Scam"] == 1
    assert stats["source_counts"]["src_a"] == 2
    assert stats["source_counts"]["src_b"] == 1
    
    # Risk stats
    assert stats["risk_stats"]["min"] == 60.0
    assert stats["risk_stats"]["max"] == 95.0
    assert stats["risk_stats"]["mean"] == 80.0
    
    # Average turns
    # rec1 = 2 turns, rec2 = 1 turn, rec3 = 3 turns. Total turns = 6. Avg = 6/3 = 2.0
    assert stats["avg_turns"] == 2.0
    
    # Average words
    # rec1 = 3 + 4 = 7 words
    # rec2 = 5 words
    # rec3 = 4 + 3 + 2 = 9 words
    # Total words = 21. Avg = 21/3 = 7.0
    assert stats["avg_words"] == 7.0
    
    # Class imbalance majority/minority = 2 / 1 = 2.0
    assert stats["imbalance"]["ratio"] == 2.0
