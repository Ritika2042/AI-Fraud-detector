from typing import Dict

RECOMMENDATIONS: Dict[str, str] = {
    "Digital Arrest Scam": "Do not join video calls. Law enforcement will never place you under digital arrest over phone or video calls. Hang up and contact official police channels.",
    "OTP Scam": "Never share OTPs, verification codes, or activation PINs with anyone. Bank employees or service representatives will never ask for them.",
    "Bank KYC Scam": "Never verify account details or share passwords over the phone. Contact your bank directly through their official numbers.",
    "UPI Scam": "Do not enter your UPI PIN to receive money. A UPI PIN is only required to send money or check your balance.",
    "QR Code Scam": "Scanning a QR code means you are sending money, not receiving it. Do not scan QR codes sent by unknown contacts.",
    "Investment Scam": "Avoid schemes promising guaranteed high returns or quick trading profits. Check regulatory registrations of financial advisors.",
    "Loan Scam": "Legitimate lenders do not ask for upfront processing fees or refundable deposits before disbursing a loan.",
    "Romance Scam": "Do not send money or gifts to people you have only met online. Verify their identity independently.",
    "Tech Support Scam": "Do not install remote access apps like AnyDesk or TeamViewer. Microsoft or your bank will never request remote access to fix issues.",
    "Courier Scam": "Do not pay customs duties or clearance fees to private accounts. Track packages only through official carrier websites.",
    "Electricity Bill Scam": "SMS or WhatsApp messages warning of power disconnection are fraudulent. Pay utility bills only via official portals.",
    "Insurance Scam": "Do not pay registration or documentation fees to claim matured insurance policies or bonuses.",
    "Crypto Scam": "Do not transfer cryptocurrency to unknown wallets or verify wallets on unauthorized platforms.",
    "Safe": "No scam detected. Keep your personal and financial information secure.",
    "Unknown Scam": "Potential scam detected. Do not share personal details, scan QR codes, or transfer money. Hang up immediately."
}

class RecommendationEngine:
    """Returns category-specific safety advice for scam warnings."""

    def __init__(self, recommendations: Dict[str, str] = RECOMMENDATIONS):
        self.recommendations = recommendations

    def get_recommendation(self, predicted_category: str) -> str:
        """Returns safety advice tailored to the predicted category."""
        return self.recommendations.get(
            predicted_category,
            "Do not share sensitive credentials, bank details, OTPs, or make upfront payments."
        )
