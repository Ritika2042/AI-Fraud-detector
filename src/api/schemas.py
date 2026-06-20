from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator

class PredictionRequest(BaseModel):
    """Schema for a single conversation transcript classification request."""
    conversation: str = Field(
        ...,
        description="The conversation transcript to classify.",
        json_schema_extra={"example": "Hello, I am calling from CBI. Your Aadhaar card has been linked to a money laundering case. You are under digital arrest."}
    )

    @field_validator("conversation")
    @classmethod
    def conversation_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Conversation cannot be empty or consisting only of whitespaces.")
        if len(v.strip()) < 5:
            raise ValueError("Conversation is too short (minimum 5 characters).")
        return v

class BatchPredictionRequest(BaseModel):
    """Schema for batch conversation transcript classification requests."""
    conversations: List[str] = Field(
        ...,
        description="A list of conversation transcripts to classify.",
        json_schema_extra={"example": [
            "Please scan this QR code to receive the payment for your laptop.",
            "Hey dear, I am sending you a very expensive watch from London. You just need to pay the customs fee of 15000 rupees."
        ]}
    )

    @field_validator("conversations")
    @classmethod
    def list_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("List of conversations cannot be empty.")
        for idx, item in enumerate(v):
            if not item.strip():
                raise ValueError(f"Conversation at index {idx} cannot be empty.")
            if len(item.strip()) < 5:
                raise ValueError(f"Conversation at index {idx} is too short (minimum 5 characters).")
        return v

class PredictionResponse(BaseModel):
    """Schema for single prediction response containing classification outputs."""
    prediction: str = Field(..., description="The predicted scam category or 'Safe'.")
    confidence: float = Field(..., description="The probability score for the predicted class.")
    confidence_level: str = Field(..., description="The qualitative confidence label (Low, Medium, High).")
    risk_score: float = Field(..., description="The numeric risk score between 0 and 100.")
    risk_level: str = Field(..., description="The estimated risk level (Critical, High, Medium, Low).")
    evidence: Dict[str, Any] = Field(..., description="Dictionary of matched scam indicators and snippets.")
    top_keywords: List[Dict[str, Any]] = Field(..., description="List of top keywords and weights contributing to the prediction.")
    recommendation: str = Field(..., description="Safety recommendation advice tailored to the category.")
    
    # Backwards compatibility fields
    risk: Optional[str] = Field(None, description="Deprecated field for backwards compatibility.")
    reasons: Optional[List[str]] = Field(None, description="Deprecated field for backwards compatibility.")

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""
    results: List[PredictionResponse] = Field(..., description="List of prediction results corresponding to input conversations.")

class TrainResponse(BaseModel):
    """Schema for the retraining pipeline status response."""
    status: str = Field(..., description="The status of the training request (e.g., 'started', 'completed').")
    message: str = Field(..., description="Detailed message regarding the training process.")

class HealthResponse(BaseModel):
    """Schema for service liveness checks."""
    status: str = Field(..., description="Health status (e.g. 'ok').")
    model_loaded: bool = Field(..., description="Whether the classifier model is loaded and ready.")
    project: str = Field(..., description="The name of the project.")
