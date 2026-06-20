import re
from typing import Dict, Any, List, Tuple

# Configuration mapping evidence categories to their respective pattern lexicons
EVIDENCE_CONFIG: Dict[str, Dict[str, Any]] = {
    "authority_impersonation": {
        "patterns": [
            r"\b(?:cbi|police|crime branch|customs|trai|telecom authority|rbi|supreme court|court office|notary public)\b",
            r"\b(?:officer|inspector|commissioner|magistrate|judge|bank manager|agent)\s+[a-za-z]+",
            r"\bcalling from\s+(?:cbi|police|customs|hq|headquarters|crime branch|bank hq)\b"
        ],
        "explanation": "Posing as law enforcement, government officials, regulatory bodies, or financial managers to establish false authority."
    },
    "otp_request": {
        "patterns": [
            r"\b(?:otp|one time password|verification code|security code|activation pin|6 digit pin)\b",
            r"\b(?:tell me|share|read out|sent to your phone|provide the)\s+(?:otp|code|pin)\b",
            r"\b(?:verification|cancel|confirm)\s+(?:otp|code)\b"
        ],
        "explanation": "Requesting one-time validation codes (OTPs) to bypass authentication checks or authorize transactions."
    },
    "urgency": {
        "patterns": [
            r"\b(?:immediately|urgently|right now|quick|fast|without delay|at once)\b",
            r"\b(?:within|in the next)\s+(?:2|5|10|24)\s+(?:hours|minutes|mins)\b",
            r"\b(?:blocked by tonight|expires today|deactivated today|pay today|act today)\b",
            r"\b(?:last warning|final notice|immediate action required)\b"
        ],
        "explanation": "Using time-pressure tactics to prevent the victim from verifying details or thinking logically."
    },
    "threat": {
        "patterns": [
            r"\b(?:jail|prison|imprisonment|lockup|custody)\b",
            r"\b(?:arrest warrant|court warrant|police action|legal actions?|file an fir|prosecution)\b",
            r"\b(?:blocked permanently|suspended indefinitely|deactivate your SIM|blacklist)\b",
            r"\b(?:face penalties|legal consequences|consequences)\b"
        ],
        "explanation": "Coercing cooperation by threatening legal action, jail time, fines, or loss of access to services."
    },
    "fear_tactics": {
        "patterns": [
            r"\b(?:money laundering|drug smuggling|illegal passport|suspicious parcel|contraband|illegal financial transaction)\b",
            r"\b(?:digital arrest|supervision acknowledgement)\b",
            r"\b(?:seize your property|scrutiny of properties|freeze all assets)\b",
            r"\b(?:damaged reputation|inform your family|disclose to relatives)\b"
        ],
        "explanation": "Injecting fear by fabricating extreme charges, placing under artificial 'digital arrest', or threatening asset seizure."
    },
    "payment_request": {
        "patterns": [
            r"\b(?:transfer|deposit|pay|wire|send)\s+(?:\w+\s+)?(?:money|funds|rupees|dollars|usdt|bitcoin)\b",
            r"\b(?:processing fee|security deposit|refundable charge|customs clearance|documentation charge|tax clearance)\b",
            r"\b(?:pay the customs|deposit the fee|refundable capital)\b"
        ],
        "explanation": "Directing the victim to send money, processing fees, customs duties, or deposits to a provided account."
    },
    "remote_access_request": {
        "patterns": [
            r"\b(?:anydesk|teamviewer|rustdesk|zoom|screen share|share screen|remote control)\b",
            r"\b(?:install|download)\s+(?:anydesk|app|support app|remote support)\b",
            r"\b(?:grant access|allow connection|share desktop)\b"
        ],
        "explanation": "Prompting the victim to install remote-desktop software to gain control over their phone or bank accounts."
    },
    "bank_account_verification": {
        "patterns": [
            r"\b(?:verify\s+(?:\w+\s+)?account|security\s+(?:\w+\s+)?account|rbi\s+(?:\w+\s+)?verification|rbi\s+(?:\w+\s+)?account|safe\s+(?:\w+\s+)?account|central\s+(?:\w+\s+)?verification)\b",
            r"\b(?:card number|cvv|expiry date|netbanking password|login credentials|atm pin)\b",
            r"\b(?:liquidate|withdraw your fds|close fixed deposit)\b"
        ],
        "explanation": "Asking for highly sensitive credentials or instructing the victim to move assets into 'secure RBI validation accounts'."
    },
    "qr_code_request": {
        "patterns": [
            r"\b(?:scan|qr code|scan this|scan the code|barcode)\b",
            r"\b(?:scan to receive|receive payment|claim cashback|scan voucher)\b",
            r"\b(?:enter your upi pin|upi password|google pay pin|phonepe pin)\b"
        ],
        "explanation": "Tricking the victim into scanning a QR code and typing their PIN under the false premise that they are receiving money."
    }
}

class EvidenceExtractor:
    """Extracts scam markers and outputs explanations with local sentence triggers and confidence ratings."""

    def __init__(self, config: Dict[str, Dict[str, Any]] = EVIDENCE_CONFIG):
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
        and scores confidence for each categories.
        """
        sentences = self.split_sentences(text)
        analysis_results = {}
        scam_score_accumulator = 0.0

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

            # Determine confidence score based on matches
            if matched_patterns_count == 0:
                confidence = 0.0
            elif matched_patterns_count == 1:
                confidence = 0.70
            elif matched_patterns_count == 2:
                confidence = 0.90
            else:
                confidence = 0.98

            detected = len(matching_snippets) > 0
            
            # Accumulate overall scam indicator density
            if detected:
                scam_score_accumulator += confidence

            analysis_results[category] = {
                "detected": detected,
                "confidence": confidence,
                "snippet": matching_snippets[0] if detected else None,
                "all_snippets": matching_snippets,
                "explanation": config["explanation"]
            }

        # Calculate overall evidence-based scam probability (capped at 1.0)
        # Having multiple distinct indicators increases overall fraud likelihood
        active_indicators = sum(1 for res in analysis_results.values() if res["detected"])
        if active_indicators == 0:
            overall_scam_prob = 0.0
        elif active_indicators == 1:
            overall_scam_prob = 0.45
        elif active_indicators == 2:
            overall_scam_prob = 0.85
        else:
            overall_scam_prob = 0.98

        return {
            "evidence_scam_probability": overall_scam_prob,
            "evidence_signals_count": active_indicators,
            "evidence": analysis_results
        }
