import re
from typing import Dict, Any, List

EVIDENCE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "otp_request": {
        "patterns": [
            r"\b(?:otp|one time password|verification code|activation pin|6 digit pin)\b",
            r"\b(?:tell me|share|read out|sent to your phone|provide the)\s+(?:otp|code|pin)\b",
            r"\b(?:verification|cancel|confirm)\s+(?:otp|code)\b"
        ],
        "explanation": "Requesting one-time validation codes (OTPs) to bypass authentication checks or authorize transactions."
    },
    "upi_pin_request": {
        "patterns": [
            r"\b(?:upi pin|upi password|google pay pin|phonepe pin|enter pin|enter upi)\b",
            r"\b(?:enter your|type your|share your)\s+(?:upi pin|pin)\b",
            r"\b(?:gpay pin|phonepe pin|paytm pin)\b"
        ],
        "explanation": "Requesting UPI PIN which is used exclusively for authorizing money transfers from the victim's account."
    },
    "qr_code_request": {
        "patterns": [
            r"\b(?:scan|qr code|scan the code|scan this|barcode)\b",
            r"\b(?:scan to receive|receive payment|claim cashback|scan voucher)\b",
            r"\b(?:scan the qr|qr scan)\b"
        ],
        "explanation": "Directing the victim to scan a QR code under the pretext of receiving money, which actually triggers a debit."
    },
    "bank_kyc": {
        "patterns": [
            r"\b(?:kyc|know your customer|update details|verify\s+(?:\w+\s+)?account|verify bank account|pan card update)\b",
            r"\b(?:kyc verification|account\s+(?:\w+\s+)?blocked|update bank record)\b",
            r"\b(?:suspension of account|verify netbanking|card number|cvv|netbanking password)\b"
        ],
        "explanation": "Claiming the victim's bank account or KYC is suspended to trick them into verifying bank credentials."
    },
    "aadhaar_pan_mentions": {
        "patterns": [
            r"\b(?:aadhaar|pan number|pan card|aadhaar card|identity proof)\b",
            r"\b(?:aadhaar verification|link your aadhaar|link pan)\b",
            r"\b(?:national identity card|uidai)\b"
        ],
        "explanation": "Requesting or mentioning Aadhaar or PAN identity cards to establish credibility or verify records."
    },
    "police_cbi_impersonation": {
        "patterns": [
            r"\b(?:police|cbi|crime branch|telecom authority|trai|supreme court|officer calling)\b",
            r"\b(?:officer|inspector|commissioner|magistrate|judge|bank manager|agent)\s+[a-z]+",
            r"\bcalling from\s+(?:cbi|police|customs|hq|headquarters|crime branch)\b"
        ],
        "explanation": "Posing as law enforcement, government officials, or regulatory bodies to establish false authority."
    },
    "digital_arrest": {
        "patterns": [
            r"\b(?:digital arrest|stay online|under surveillance|video call monitoring)\b",
            r"\b(?:do not hang up|stay on camera|skype calling|digital custody)\b",
            r"\b(?:video surveillance|confidential investigation)\b"
        ],
        "explanation": "Placing the victim under 'digital arrest' via constant video surveillance to isolate and coerce them."
    },
    "money_transfer_request": {
        "patterns": [
            r"\b(?:transfer money|deposit funds|send money|wire transfer|pay now)\b",
            r"\b(?:transfer|deposit|pay|wire|send)\s+(?:\w+\s+)?(?:money|funds|rupees|dollars|usdt|bitcoin)\b",
            r"\b(?:account number|account details|beneficiary)\b"
        ],
        "explanation": "Directing the victim to send or transfer money or funds to a specified beneficiary bank account."
    },
    "cryptocurrency_mentions": {
        "patterns": [
            r"\b(?:crypto|usdt|bitcoin|wallet address|staking|binance)\b",
            r"\b(?:cryptocurrency|blockchain|trust wallet|metamask|digital assets)\b",
            r"\b(?:buy crypto|transfer usdt)\b"
        ],
        "explanation": "Demanding payments or promoting investments in untraceable cryptocurrencies."
    },
    "urgency_phrases": {
        "patterns": [
            r"\b(?:immediately|urgently|right now|without delay|within 5 minutes|expires today)\b",
            r"\b(?:within|in the next)\s+(?:2|5|10|24)\s+(?:hours|minutes|mins)\b",
            r"\b(?:blocked by tonight|expires today|deactivated today|pay today|act today)\b",
            r"\b(?:last warning|final notice|immediate action required)\b"
        ],
        "explanation": "Creating fake time urgency to force the victim into making decisions without thinking or verifying."
    },
    "threat_language": {
        "patterns": [
            r"\b(?:jail|prison|arrest warrant|blacklist|prosecution|face penalties|legal action)\b",
            r"\b(?:file an fir|police action|legal actions?|prosecution|confiscate)\b",
            r"\b(?:blocked permanently|suspended indefinitely|deactivate your SIM)\b"
        ],
        "explanation": "Threatening jail time, arrest, service deactivation, or legal actions to coerce compliance."
    },
    "gift_customs_request": {
        "patterns": [
            r"\b(?:gift|customs clearance|customs duty|customs charge|parcel arrived|contraband)\b",
            r"\b(?:customs officer|package trapped|clearing charge|parcel scan|customs fee)\b",
            r"\b(?:valuable package|foreign parcel)\b"
        ],
        "explanation": "Claiming a valuable parcel or gift is stuck at customs and demanding clearance fees."
    },
    "loan_processing_fee": {
        "patterns": [
            r"\b(?:processing fee|documentation fee|registration fee|refundable deposit|loan approval fee)\b",
            r"\b(?:refundable capital|security deposit|loan charges|pay processing)\b",
            r"\b(?:pre-payment fee|disbursement charge)\b"
        ],
        "explanation": "Demanding upfront processing or documentation fees as a prerequisite for loan disbursement."
    },
    "investment_promises": {
        "patterns": [
            r"\b(?:investment|returns|double money|guaranteed profits|high yield|premium signals)\b",
            r"\b(?:invest|quick profits|guaranteed return|trading profits|exclusive tips)\b",
            r"\b(?:passive income|stock group|professor trading)\b"
        ],
        "explanation": "Promising unrealistic, guaranteed investment returns or trading profits in short timeframes."
    }
}

class EvidenceExtractor:
    """Extracts scam markers and outputs explanations with local sentence triggers and confidence ratings."""

    def __init__(self, config: Dict[str, Dict[str, Any]] = EVIDENCE_PATTERNS):
        # Pre-compile patterns for speed
        self.evidence_categories = {}
        for category, data in config.items():
            compiled_patterns = [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            self.evidence_categories[category] = {
                "patterns": compiled_patterns,
                "explanation": data["explanation"]
            }

    def split_sentences(self, text: str) -> List[str]:
        """Splits raw text into sentences while ignoring minor spacing noise."""
        if not text:
            return []
        # Split on sentence terminals followed by space or newline
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_evidence(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the conversation transcript, extracts evidence snippets,
        and scores confidence for each of the 14 categories.
        """
        sentences = self.split_sentences(text)
        analysis_results = {}
        active_indicators = 0

        for category, config in self.evidence_categories.items():
            detected = False
            matching_snippets = []
            matched_patterns_count = 0
            
            # Check pattern match across all sentences
            for sentence in sentences:
                sentence_matched = False
                for pattern in config["patterns"]:
                    if pattern.search(sentence):
                        if sentence not in matching_snippets:
                            matching_snippets.append(sentence)
                        sentence_matched = True
                if sentence_matched:
                    matched_patterns_count += 1

            detected = len(matching_snippets) > 0
            if detected:
                active_indicators += 1
                # Determine confidence rating based on matches
                if matched_patterns_count == 1:
                    confidence = 0.70
                elif matched_patterns_count == 2:
                    confidence = 0.90
                else:
                    confidence = 0.98
            else:
                confidence = 0.0

            analysis_results[category] = {
                "detected": detected,
                "confidence": confidence,
                "snippet": matching_snippets[0] if detected else None,
                "all_snippets": matching_snippets,
                "explanation": config["explanation"]
            }

        return {
            "evidence_signals_count": active_indicators,
            "evidence": analysis_results
        }
