import logging
from typing import Set, Dict, List
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from src.logger import get_logger

logger = get_logger()

# Set of allowed categories for the classification model
ALLOWED_LABELS: Set[str] = {
    "Safe",
    "Digital Arrest Scam",
    "OTP Scam",
    "Bank KYC Scam",
    "Investment Scam",
    "Loan Scam",
    "QR Code Scam",
    "Job Scam",
    "Romance Scam",
    "Crypto Scam"
}

class ConversationSchema(BaseModel):
    """Pydantic schema to validate individual inference inputs or single records."""
    text: str = Field(..., description="The conversation transcript text.")
    label: str = Field(None, description="The ground-truth classification label.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Transcript must be a string.")
        if not v.strip():
            raise ValueError("Transcript cannot be empty.")
        if len(v.strip()) < 5:
            raise ValueError("Transcript is too short to perform classification.")
        return v

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v is not None and v not in ALLOWED_LABELS:
            raise ValueError(f"Label '{v}' must be one of the allowed classes: {ALLOWED_LABELS}")
        return v

class DataValidator:
    """Class to perform data validation checks on pandas DataFrames."""

    def __init__(self, allowed_labels: Set[str] = ALLOWED_LABELS):
        self.allowed_labels = allowed_labels

    def validate_training_df(self, df: pd.DataFrame) -> bool:
        """
        Validates training data schema and quality checks.
        Returns True if the dataset is valid, False otherwise.
        """
        # 1. Check required columns
        required_cols = {"text", "label"}
        if not required_cols.issubset(df.columns):
            logger.error(f"Data validation failed: missing columns. Expected {required_cols}, found {list(df.columns)}")
            return False

        # 2. Check for empty dataframe
        if len(df) == 0:
            logger.error("Data validation failed: DataFrame contains zero records.")
            return False

        # 3. Check for null or empty values
        null_texts = df["text"].isna().sum()
        null_labels = df["label"].isna().sum()
        
        if null_texts > 0:
            logger.warning(f"Data validation warning: {null_texts} rows have empty 'text'. These will be ignored.")
        if null_labels > 0:
            logger.warning(f"Data validation warning: {null_labels} rows have empty 'label'. These will be ignored.")

        # 4. Check for illegal label names
        # Drop null values first to prevent error in unique labels check
        cleaned_labels_df = df.dropna(subset=["label"])
        unique_labels = set(cleaned_labels_df["label"].unique())
        invalid_labels = unique_labels - self.allowed_labels
        
        if invalid_labels:
            logger.error(f"Data validation failed: invalid class labels {invalid_labels}. Must be in {self.allowed_labels}")
            return False

        # 5. Check class balance and coverage
        missing_classes = self.allowed_labels - unique_labels
        if missing_classes:
            logger.warning(f"Data validation warning: missing classes in training set: {missing_classes}")

        class_counts: Dict[str, int] = cleaned_labels_df["label"].value_counts().to_dict()
        logger.info(f"Training dataset class distribution: {class_counts}")

        # Warn if any class has less than 5 samples (K-Fold requires at least few samples)
        for label, count in class_counts.items():
            if count < 5:
                logger.warning(f"Data validation warning: class '{label}' has only {count} samples. Min recommended is 5.")

        return True
