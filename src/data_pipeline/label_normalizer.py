import re
from enum import Enum
from typing import List, Dict, Union

class ScamTaxonomy(str, Enum):
    SAFE = "Safe"
    DIGITAL_ARREST = "Digital Arrest Scam"
    BANK_KYC = "Bank KYC Scam"
    UPI = "UPI Scam"
    QR_CODE = "QR Code Scam"
    INVESTMENT = "Investment Scam"
    LOAN = "Loan Scam"
    JOB = "Job Scam"
    ROMANCE = "Romance Scam"
    TECH_SUPPORT = "Tech Support Scam"
    COURIER = "Courier Scam"
    ELECTRICITY_BILL = "Electricity Bill Scam"
    INSURANCE = "Insurance Scam"
    CRYPTO = "Crypto Scam"
    MARKETING = "Marketing Scam"
    LOTTERY = "Lottery/Prize Scam"
    UNKNOWN_SCAM = "Unknown Scam"
    OTP = "OTP Scam"

class LabelNormalizer:
    """
    Normalizes datasets into a unified scam taxonomy using rule-based keyword mapping.
    Also provides risk scoring and behavior tag detection based on conversation text.
    Uses regex word boundaries to avoid false-positive substring matches.
    """
    def __init__(self, custom_rules: Dict[ScamTaxonomy, List[str]] = None):
        # Centralized matching keyword rules (case-insensitive checks)
        self.rules: Dict[ScamTaxonomy, List[str]] = {
            ScamTaxonomy.BANK_KYC: [
                "kyc", 
                "bank account", 
                "rbi", 
                "account blocked"
            ],
            ScamTaxonomy.DIGITAL_ARREST: [
                "police", 
                "cbi", 
                "aadhaar", 
                "money laundering", 
                "digital arrest"
            ],
            ScamTaxonomy.INVESTMENT: [
                "invest", 
                "crypto", 
                "profit", 
                "double money"
            ],
            ScamTaxonomy.QR_CODE: [
                "qr", 
                "scan code"
            ],
            ScamTaxonomy.LOTTERY: [
                "congratulations", 
                "win", 
                "prize", 
                "lottery", 
                "free entry"
            ],
            ScamTaxonomy.MARKETING: [
                "discount", 
                "offer", 
                "promotion"
            ],
            ScamTaxonomy.JOB: [
                "whatsapp job", 
                "part-time work", 
                "earn money online"
            ],
            ScamTaxonomy.ROMANCE: [
                "love", 
                "dear", 
                "darling", 
                "romance"
            ],
            ScamTaxonomy.TECH_SUPPORT: [
                "tech support", 
                "technical support", 
                "virus", 
                "malware"
            ],
            ScamTaxonomy.COURIER: [
                "courier", 
                "package", 
                "delivery", 
                "customs fee"
            ],
            ScamTaxonomy.ELECTRICITY_BILL: [
                "electricity bill", 
                "power bill", 
                "electricity connection"
            ],
            ScamTaxonomy.INSURANCE: [
                "insurance", 
                "policy", 
                "premium"
            ],
            ScamTaxonomy.LOAN: [
                "loan", 
                "borrow", 
                "interest rate"
            ],
            ScamTaxonomy.UPI: [
                "upi", 
                "gpay", 
                "phonepe", 
                "upi pin"
            ],
            ScamTaxonomy.CRYPTO: [
                "crypto", 
                "bitcoin", 
                "usdt", 
                "wallet address"
            ]
        }
        
        # Override or add custom rules if provided
        if custom_rules:
            for label, keywords in custom_rules.items():
                self.rules[label] = [kw.lower() for kw in keywords]

        # Ensure OTP matches are checked and correctly defined (this must be set!)
        # Requirement list also specified OTP keywords explicitly.
        if ScamTaxonomy.OTP not in self.rules:
            self.rules[ScamTaxonomy.OTP] = [
                "otp", 
                "verification code", 
                "one time password"
            ]

    def register_rule(self, label: ScamTaxonomy, keywords: List[str]) -> None:
        """
        Dynamically registers or extends matching keywords for a specific ScamTaxonomy label.
        """
        if label not in self.rules:
            self.rules[label] = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.rules[label]:
                self.rules[label].append(kw_lower)

    def _has_word_match(self, text: str, keyword: str) -> bool:
        """Helper to find case-insensitive full word matches using regex boundaries."""
        pattern = rf"\b{re.escape(keyword.lower())}\b"
        return bool(re.search(pattern, text.lower()))

    def normalize(self, text: str, current_label: str = None) -> ScamTaxonomy:
        """
        Normalizes a given transcript text and/or an existing raw label into the ScamTaxonomy.
        
        Args:
            text: The conversation transcript text.
            current_label: Optional current raw label associated with the text.
            
        Returns:
            The matched ScamTaxonomy enum value.
        """
        # 1. Clean and normalize input
        text_lower = (text or "").lower().strip()
        label_lower = (current_label or "").lower().strip()
        
        # 2. Check direct exact match first to one of our taxonomy labels
        for taxonomy_member in ScamTaxonomy:
            if label_lower == taxonomy_member.value.lower():
                return taxonomy_member
                
        # 3. Check if the current label matches known safe representations
        if label_lower in ["safe", "ham", "legitimate", "0"]:
            return ScamTaxonomy.SAFE
            
        # 4. Use rule-based keyword matching on the conversation text
        for label, keywords in self.rules.items():
            for keyword in keywords:
                if self._has_word_match(text_lower, keyword):
                    return label
                    
        # 5. If it was flagged as spam/scam/fraud but didn't match any keyword, map to Unknown Scam
        if label_lower in ["scam", "spam", "spam/scam", "fraudulent", "1"]:
            return ScamTaxonomy.UNKNOWN_SCAM
            
        # Default fallback
        return ScamTaxonomy.UNKNOWN_SCAM

    def get_risk_score(self, label: ScamTaxonomy) -> float:
        """
        Retrieves the standard risk score (0-100) associated with a given taxonomy label.
        """
        scores = {
            ScamTaxonomy.SAFE: 0.0,
            ScamTaxonomy.MARKETING: 30.0,
            ScamTaxonomy.LOTTERY: 50.0,
            ScamTaxonomy.JOB: 60.0,
            ScamTaxonomy.BANK_KYC: 80.0,
            ScamTaxonomy.OTP: 85.0,
            ScamTaxonomy.DIGITAL_ARREST: 98.0,
            ScamTaxonomy.UPI: 85.0,
            ScamTaxonomy.QR_CODE: 85.0,
            ScamTaxonomy.INVESTMENT: 75.0,
            ScamTaxonomy.LOAN: 70.0,
            ScamTaxonomy.ROMANCE: 65.0,
            ScamTaxonomy.TECH_SUPPORT: 80.0,
            ScamTaxonomy.COURIER: 80.0,
            ScamTaxonomy.ELECTRICITY_BILL: 80.0,
            ScamTaxonomy.INSURANCE: 80.0,
            ScamTaxonomy.CRYPTO: 90.0,
            ScamTaxonomy.UNKNOWN_SCAM: 70.0
        }
        return scores.get(label, 70.0)

    def detect_behavior_tags(self, text: str) -> Dict[str, bool]:
        """
        Detects binary behavior flags indicating specific scam markers based on text keyword matching.
        """
        text_lower = (text or "").lower()
        tag_keywords = {
            "authority_impersonation": ["police", "cbi", "officer", "government", "rbi", "inspector", "agent", "customs", "cops"],
            "urgency": ["immediately", "urgent", "quick", "fast", "limited time", "suspend", "blocks", "suspended", "today", "now", "warning"],
            "fear": ["arrest", "jail", "legal action", "court", "frozen", "blocked", "criminal", "penalty", "imprisonment", "prosecuted"],
            "money_request": ["fee", "pay", "transfer", "deposit", "charge", "payment", "send", "cost", "rupees", "dollars", "funds"],
            "otp_request": ["otp", "verification code", "one time password", "pin", "code"],
            "bank_details_request": ["kyc", "bank account", "card details", "cvv", "account number", "credit card", "debit card"],
            "qr_request": ["qr", "scan code", "scan qr", "scan to pay"],
            "keep_secret": ["secret", "don't tell", "keep quiet", "do not share", "confidential", "silent", "tell no one"],
            "remote_access_request": ["anydesk", "teamviewer", "remote access", "screen share", "download app", "install app"]
        }
        
        return {
            tag: any(self._has_word_match(text_lower, kw) for kw in kws)
            for tag, kws in tag_keywords.items()
        }
