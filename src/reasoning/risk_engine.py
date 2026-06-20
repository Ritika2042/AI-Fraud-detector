from typing import Dict, Any

EVIDENCE_WEIGHTS: Dict[str, float] = {
    "upi_pin_request": 20.0,
    "digital_arrest": 20.0,
    "otp_request": 15.0,
    "qr_code_request": 15.0,
    "police_cbi_impersonation": 15.0,
    "bank_kyc": 10.0,
    "money_transfer_request": 10.0,
    "threat_language": 10.0,
    "gift_customs_request": 10.0,
    "loan_processing_fee": 10.0,
    "investment_promises": 10.0,
    "aadhaar_pan_mentions": 5.0,
    "cryptocurrency_mentions": 5.0,
    "urgency_phrases": 5.0
}

class RiskEngine:
    """Computes risk scores and risk levels based on extracted evidence indicators."""

    def __init__(self, weights: Dict[str, float] = EVIDENCE_WEIGHTS):
        self.weights = weights

    def compute_risk(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes 0-100 risk score and maps to a risk level.
        `evidence` is the "evidence" field from EvidenceExtractor.
        """
        total_weight = 0.0
        
        for category, details in evidence.items():
            if details.get("detected", False):
                weight = self.weights.get(category, 0.0)
                total_weight += weight

        # Cap the risk score at 100.0
        risk_score = min(100.0, total_weight)
        
        # Determine risk level
        if risk_score < 30.0:
            risk_level = "Low"
        elif risk_score < 70.0:
            risk_level = "Medium"
        elif risk_score < 90.0:
            risk_level = "High"
        else:
            risk_level = "Critical"

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level
        }
