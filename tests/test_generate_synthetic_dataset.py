import pytest
from scripts.generate_synthetic_dataset import ScamConversationGenerator
from src.data_pipeline.label_normalizer import ScamTaxonomy

def test_generate_turns_format():
    generator = ScamConversationGenerator()
    categories = [
        ScamTaxonomy.DIGITAL_ARREST,
        ScamTaxonomy.OTP,
        ScamTaxonomy.BANK_KYC,
        ScamTaxonomy.UPI,
        ScamTaxonomy.QR_CODE,
        ScamTaxonomy.INVESTMENT,
        ScamTaxonomy.LOAN,
        ScamTaxonomy.ROMANCE,
        ScamTaxonomy.TECH_SUPPORT,
        ScamTaxonomy.COURIER,
        ScamTaxonomy.ELECTRICITY_BILL,
        ScamTaxonomy.INSURANCE,
        ScamTaxonomy.CRYPTO
    ]

    for cat in categories:
        for split in ["train", "test", "validation"]:
            turns = generator.generate_turns(cat, split)
            
            # Verify turns exist and are within the specified range (2-10 turns)
            assert len(turns) >= 2
            assert len(turns) <= 10
            
            # Verify formats of speakers and text
            for turn in turns:
                assert "speaker" in turn
                assert "text" in turn
                assert len(turn["speaker"]) > 0
                assert len(turn["text"]) > 0

def test_generate_dataset_structure():
    generator = ScamConversationGenerator()
    count_per_class = 2
    
    for split in ["train", "test", "validation"]:
        records = generator.generate_split_dataset(count_per_class, split)
        
        # 14 classes * 2 records = 28 records
        assert len(records) == 28
        
        for rec in records:
            assert "id" in rec
            assert "language" in rec
            assert "source" in rec
            assert "conversation" in rec
            assert "label" in rec
            assert "risk_score" in rec
            assert "tags" in rec
            
            assert rec["language"] == "en"
            assert rec["source"] == "synthetic"
            assert isinstance(rec["conversation"], list)
            assert len(rec["conversation"]) >= 2
            
            # Verify label matches one of the 13 categories
            assert rec["label"] in [t.value for t in ScamTaxonomy]
            
            # Verify risk score is on the 0-100 scale
            assert 0.0 <= rec["risk_score"] <= 100.0
            
            # Verify behavior tags
            tags = rec["tags"]
            required_tags = [
                "authority_impersonation",
                "urgency",
                "fear",
                "money_request",
                "otp_request",
                "bank_details_request",
                "qr_request",
                "keep_secret",
                "remote_access_request"
            ]
            for tag in required_tags:
                assert tag in tags
                assert isinstance(tags[tag], bool)
