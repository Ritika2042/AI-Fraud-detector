import pandas as pd
from scripts.convert_datasets import DatasetConverter

def test_detect_dataset_type():
    converter = DatasetConverter()
    
    # 1. Test Fake Job Posting signature detection
    df_job = pd.DataFrame(columns=["Title", "Description", "Requirements", "telecommuting", "Fraudulent"])
    assert converter.detect_dataset_type(df_job) == "fake_job_posting"
    
    # 2. Test Email Spam signature detection
    df_spam = pd.DataFrame(columns=["Text", "Spam"])
    assert converter.detect_dataset_type(df_spam) == "email_spam"
    
    df_sms = pd.DataFrame(columns=["v1", "v2"])
    assert converter.detect_dataset_type(df_sms) == "email_spam"
    
    # 3. Test Unified Scam signature detection
    df_unified = pd.DataFrame(columns=["conversation", "Label"])
    assert converter.detect_dataset_type(df_unified) == "unified_scam"
    
    # 4. Test Unknown signature detection
    df_unknown = pd.DataFrame(columns=["Id", "Age", "Salary"])
    assert converter.detect_dataset_type(df_unknown) == "unknown"

def test_convert_record_safe():
    converter = DatasetConverter()
    raw_text = "Hi Mom, I will reach home by 7 PM tonight. Please don't wait."
    record = converter.convert_record(raw_text, "Safe", "test_source_dataset")
    
    assert record.label == "Safe"
    assert record.risk_score == 0.0
    assert record.source == "test_source_dataset"
    assert len(record.conversation) == 1
    assert record.conversation[0].speaker == "Sender"
    assert "reach home" in record.conversation[0].text
    assert record.tags["money_request"] is False

def test_convert_record_scam():
    converter = DatasetConverter()
    text = (
        "[Officer]: I am calling from the CBI head office.\n"
        "[Victim]: What happened?\n"
        "[Officer]: Your card was linked to a money laundering case. Transfer 50000 rupees to our safe verification account immediately or face imprisonment."
    )
    record = converter.convert_record(text, "1", "test_source_dataset")
    
    assert record.label == "Digital Arrest Scam"
    assert record.risk_score == 98.0
    assert len(record.conversation) == 3
    assert record.conversation[0].speaker == "Officer"
    assert record.conversation[1].speaker == "Victim"
    
    # Verify indicator tags extract correctly
    assert record.tags["authority_impersonation"] is True
    assert record.tags["urgency"] is True
    assert record.tags["fear"] is True
    assert record.tags["money_request"] is True
    assert record.tags["bank_details_request"] is False
