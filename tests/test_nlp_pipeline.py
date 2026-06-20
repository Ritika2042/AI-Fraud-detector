from src.data_pipeline.nlp_pipeline import NLPPreprocessingPipeline

def test_nlp_pipeline_basic_cleaning():
    pipeline = NLPPreprocessingPipeline()
    
    # Check currency conversion and normalization
    raw_text = "Pay ₹500 or Rs. 1000 now."
    cleaned = pipeline.clean_and_normalize(raw_text)
    assert "rupees" in cleaned
    assert "₹" not in cleaned
    assert "rs" not in cleaned

def test_nlp_pipeline_pii_masking():
    pipeline = NLPPreprocessingPipeline()
    
    raw_text = "Send credentials to agent-jack@secure-bank.org or call +91-9876543210. IP is 192.168.1.1."
    masked, counts = pipeline.mask_pii(raw_text)
    
    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked
    assert "[IP_ADDRESS]" in masked
    assert counts["EMAIL"] == 1
    assert counts["PHONE"] == 1
    assert counts["IP_ADDRESS"] == 1

def test_nlp_pipeline_speaker_separation():
    pipeline = NLPPreprocessingPipeline()
    
    transcript = (
        "[Officer Kumar]: You are placed under digital arrest.\n"
        "Please cooperate.\n"
        "Speaker 2: Why? I have not done anything wrong."
    )
    
    turns = pipeline.separate_speakers(transcript)
    
    assert len(turns) == 2
    assert turns[0]["speaker"] == "Officer Kumar"
    assert "cooperate" in turns[0]["text"]
    assert turns[1]["speaker"] == "Speaker 2"
    assert "wrong" in turns[1]["text"]

def test_nlp_pipeline_segmentation():
    pipeline = NLPPreprocessingPipeline()
    
    turns = [
        {"speaker": "A", "text": "Turn 1"},
        {"speaker": "B", "text": "Turn 2"},
        {"speaker": "A", "text": "Turn 3"},
        {"speaker": "B", "text": "Turn 4"},
        {"speaker": "A", "text": "Turn 5"}
    ]
    
    # Window size 3, overlap 1 -> Step = 2
    # Windows: [1,2,3], [3,4,5]
    segments = pipeline.segment_conversation(turns, window_size=3, overlap=1)
    
    assert len(segments) == 2
    assert len(segments[0]) == 3
    assert segments[0][0]["text"] == "Turn 1"
    assert segments[1][0]["text"] == "Turn 3"

def test_nlp_pipeline_features():
    pipeline = NLPPreprocessingPipeline()
    
    text = "Urgently transfer the rupees immediately! Are you listening?"
    turns = [{"speaker": "Scammer", "text": text}]
    
    features = pipeline.extract_features(text, turns)
    
    assert features["urgency_score"] >= 2  # "urgently", "immediately"
    assert features["financial_score"] >= 2  # "transfer", "rupees"
    assert features["question_count"] == 1
    assert features["exclamation_count"] == 1
    assert features["speaker_count"] == 1

def test_nlp_pipeline_full_process():
    pipeline = NLPPreprocessingPipeline()
    
    raw = (
        "[Scammer]: Pay ₹50000 immediately to secure-lock@bank.com.\n"
        "[Victim]: What? Call me on 98765-43210 first."
    )
    
    result = pipeline.process(raw)
    
    assert "cleaned_transcript" in result
    assert "turns" in result
    assert "features" in result
    
    # Verify cleaning, PII masks, and features are populated
    assert "[EMAIL]" in result["cleaned_transcript"]
    assert "rupees" in result["cleaned_transcript"]
    assert len(result["turns"]) == 2
    assert result["features"]["urgency_score"] >= 1
