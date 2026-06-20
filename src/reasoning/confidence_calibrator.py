class ConfidenceCalibrator:
    """Converts raw numeric model confidence scores into qualitative labels (Low, Medium, High)."""

    def __init__(self, low_threshold: float = 0.60, high_threshold: float = 0.85):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def calibrate(self, confidence: float) -> str:
        """
        Calibrates numeric confidence (typically 0.0 to 1.0) to a rating label.
        """
        if confidence < self.low_threshold:
            return "Low"
        elif confidence < self.high_threshold:
            return "Medium"
        else:
            return "High"
