import pytest
from src.data_pipeline.label_normalizer import ScamTaxonomy, LabelNormalizer

def test_label_normalization_direct_match():
    normalizer = LabelNormalizer()
    
    # Direct taxonomy label match (case insensitive)
    assert normalizer.normalize("some transcript", "OTP Scam") == ScamTaxonomy.OTP
    assert normalizer.normalize("some transcript", "digital arrest scam") == ScamTaxonomy.DIGITAL_ARREST
    assert normalizer.normalize("some transcript", "Safe") == ScamTaxonomy.SAFE
    assert normalizer.normalize("some transcript", "safe") == ScamTaxonomy.SAFE
    assert normalizer.normalize("some transcript", "0") == ScamTaxonomy.SAFE

def test_rule_based_keyword_matching():
    normalizer = LabelNormalizer()
    
    # OTP Scam keyword matches
    assert normalizer.normalize("Your OTP is 1234. Do not share it.", "spam") == ScamTaxonomy.OTP
    assert normalizer.normalize("Please provide the verification code sent to your mobile.", "scam") == ScamTaxonomy.OTP
    assert normalizer.normalize("What is the one time password?", "1") == ScamTaxonomy.OTP
    
    # Bank KYC Scam keyword matches
    assert normalizer.normalize("Your KYC is pending. Please update your bank details.", "spam") == ScamTaxonomy.BANK_KYC
    assert normalizer.normalize("This is the RBI department calling.", "scam") == ScamTaxonomy.BANK_KYC
    assert normalizer.normalize("Your account blocked due to suspicious activity.", "1") == ScamTaxonomy.BANK_KYC
    
    # Digital Arrest Scam keyword matches
    assert normalizer.normalize("This is the police department.", "spam") == ScamTaxonomy.DIGITAL_ARREST
    assert normalizer.normalize("You are involved in a money laundering case.", "scam") == ScamTaxonomy.DIGITAL_ARREST
    assert normalizer.normalize("We have placed you under digital arrest.", "1") == ScamTaxonomy.DIGITAL_ARREST
    
    # Investment Scam keyword matches
    assert normalizer.normalize("Invest now and get 50% returns daily.", "spam") == ScamTaxonomy.INVESTMENT
    assert normalizer.normalize("This trading group guarantees high profit.", "scam") == ScamTaxonomy.INVESTMENT
    
    # QR Code Scam keyword matches
    assert normalizer.normalize("Scan this QR code to receive the payment.", "spam") == ScamTaxonomy.QR_CODE
    
    # Job Scam keyword matches
    assert normalizer.normalize("Earn money online with this part-time work.", "spam") == ScamTaxonomy.JOB
    
    # Lottery and Marketing keyword matches
    assert normalizer.normalize("Congratulations, you won a free lottery entry!", "spam") == ScamTaxonomy.LOTTERY
    assert normalizer.normalize("Get a 50% discount coupon code.", "spam") == ScamTaxonomy.MARKETING

def test_fallback_behavior():
    normalizer = LabelNormalizer()
    
    # Unmatched scam label -> Unknown Scam
    assert normalizer.normalize("Unrelated conversation content.", "scam") == ScamTaxonomy.UNKNOWN_SCAM
    assert normalizer.normalize("Hello there.", "spam") == ScamTaxonomy.UNKNOWN_SCAM
    assert normalizer.normalize("Random text.", "1") == ScamTaxonomy.UNKNOWN_SCAM
    
    # Unlabeled unrelated text -> Unknown Scam
    assert normalizer.normalize("Hello how are you?", None) == ScamTaxonomy.UNKNOWN_SCAM

def test_extendability_and_custom_rules():
    # Instantiate with custom rules
    custom = {
        ScamTaxonomy.UPI: ["upi pin", "gpay transfer"],
        ScamTaxonomy.ROMANCE: ["love of my life", "send gift money"]
    }
    normalizer = LabelNormalizer(custom_rules=custom)
    
    # Test custom rules
    assert normalizer.normalize("Please enter your UPI PIN to authenticate.", "scam") == ScamTaxonomy.UPI
    assert normalizer.normalize("You are the love of my life.", "scam") == ScamTaxonomy.ROMANCE
    
    # Dynamic registration extension
    normalizer.register_rule(ScamTaxonomy.CRYPTO, ["bitcoin wallet", "usdt deposit"])
    assert normalizer.normalize("Send USDT deposit to this address.", "scam") == ScamTaxonomy.CRYPTO

def test_risk_scoring():
    normalizer = LabelNormalizer()
    
    # Validate specific risk scores
    assert normalizer.get_risk_score(ScamTaxonomy.SAFE) == 0.0
    assert normalizer.get_risk_score(ScamTaxonomy.MARKETING) == 30.0
    assert normalizer.get_risk_score(ScamTaxonomy.LOTTERY) == 50.0
    assert normalizer.get_risk_score(ScamTaxonomy.JOB) == 60.0
    assert normalizer.get_risk_score(ScamTaxonomy.BANK_KYC) == 80.0
    assert normalizer.get_risk_score(ScamTaxonomy.OTP) == 85.0
    assert normalizer.get_risk_score(ScamTaxonomy.DIGITAL_ARREST) == 98.0
    
    # Fallback/Default check
    assert normalizer.get_risk_score(ScamTaxonomy.UNKNOWN_SCAM) == 70.0

def test_behavior_tag_detection():
    normalizer = LabelNormalizer()
    
    text = "We are CBI officers. Pay the processing fee immediately or go to jail. Download Anydesk to update your KYC."
    tags = normalizer.detect_behavior_tags(text)
    
    assert tags["authority_impersonation"] is True  # CBI / officer
    assert tags["urgency"] is True                  # immediately
    assert tags["fear"] is True                     # jail
    assert tags["money_request"] is True            # fee / pay
    assert tags["bank_details_request"] is True     # kyc
    assert tags["remote_access_request"] is True    # Anydesk / download app
    
    # Non-triggered tags
    assert tags["otp_request"] is False
    assert tags["qr_request"] is False
