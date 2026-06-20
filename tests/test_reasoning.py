from src.modeling.reasoning import EvidenceExtractor

def test_evidence_extractor_benign():
    extractor = EvidenceExtractor()
    text = "Hi, are we still meeting for lunch today? Let me know when you leave."
    result = extractor.extract_evidence(text)
    
    assert result["evidence_signals_count"] == 0
    assert result["evidence_scam_probability"] == 0.0
    for k, val in result["evidence"].items():
        assert val["detected"] is False
        assert val["snippet"] is None
        assert len(val["all_snippets"]) == 0

def test_evidence_extractor_scam_indicators():
    extractor = EvidenceExtractor()
    text = (
        "Hello sir, I am calling from the CBI head office. "
        "Your Aadhaar card has been linked to a money laundering case. "
        "You are placed under digital arrest immediately. "
        "If you do not cooperate you will face imprisonment. "
        "Please transfer 50000 rupees to our safe verification account now."
    )
    result = extractor.extract_evidence(text)
    
    # Verify authority impersonation
    assert result["evidence"]["authority_impersonation"]["detected"] is True
    assert "CBI" in result["evidence"]["authority_impersonation"]["snippet"]
    
    # Verify urgency is caught
    assert result["evidence"]["urgency"]["detected"] is True
    assert "immediately" in result["evidence"]["urgency"]["snippet"].lower() or "now" in result["evidence"]["urgency"]["snippet"].lower()
    
    # Verify threat/fear
    assert result["evidence"]["threat"]["detected"] is True
    assert result["evidence"]["fear_tactics"]["detected"] is True
    assert any("arrest" in s.lower() for s in result["evidence"]["fear_tactics"]["all_snippets"])
    
    # Verify payment request
    assert result["evidence"]["payment_request"]["detected"] is True
    
    # Verify bank account verification
    assert result["evidence"]["bank_account_verification"]["detected"] is True
    
    # Verify signals count is high and overall score is maximum
    assert result["evidence_signals_count"] >= 5
    assert result["evidence_scam_probability"] == 0.98

def test_sentence_splitting():
    extractor = EvidenceExtractor()
    text = "First sentence. Second sentence! What about the third? Yes."
    sentences = extractor.split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "What about the third?"
    assert sentences[3] == "Yes."
