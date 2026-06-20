import os
import sys
import re
import pickle
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to sys.path to enable importing local src modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from scripts.convert_datasets import ConvertedScamRecord
from src.config import load_config
from src.logger import get_logger

logger = get_logger()

# Define patterns for entity preservation (compiled for lowercased text)
PRESERVE_PATTERNS = [
    ("AADHAAR", re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")),
    ("PHONE", re.compile(
        r"\+?\d{1,4}[-.\s]?\(?\d{2,5}\)?[-.\s]?\d{2,5}[-.\s]?\d{2,5}(?:[-.\s]?\d{2,5})?\b|"
        r"(?:\+?91[-\s]?|0[-\s]?)?[6-9]\d{9}\b|"
        r"\b\d{10,15}\b"
    )),
    ("PAN", re.compile(r"\b[a-z]{5}\d{4}[a-z]\b")),
    ("IFSC", re.compile(r"\b[a-z]{4}0[a-z0-9]{6}\b")),
    ("UPI", re.compile(r"\b[a-z0-9.\-_]+@[a-z]{2,}\b")),
    ("OTP", re.compile(r"\b\d{4,8}\b")),
    ("CURRENCY", re.compile(r"(?:₹|\$|£|€|¥|\brs\.?|\brupees?\b|\bdollars?\b|\beuros?\b|\bpounds?\b)")),
    ("BANK", re.compile(
        r"\b(?:state\s+bank\s+of\s+india|sbi|hdfc(?:\s+bank)?|icici(?:\s+bank)?|axis(?:\s+bank)?|"
        r"punjab\s+national\s+bank|pnb|kotak(?:\s+mahindra)?(?:\s+bank)?|yes\s+bank|"
        r"indusind(?:\s+bank)?|bank\s+of\s+baroda|union\s+bank|canara(?:\s+bank)?|citibank|citi|"
        r"rbl(?:\s+bank)?|paytm(?:\s+payments\s+bank)?|phonepe|gpay|google\s+pay)\b"
    ))
]

def validate_record(record_dict: Dict[str, Any]) -> ConvertedScamRecord:
    """Validates the input record against ConvertedScamRecord Pydantic model."""
    return ConvertedScamRecord(**record_dict)

def join_turns(record: ConvertedScamRecord) -> str:
    """Joins conversation turns into one unified text string."""
    turns_text = []
    for turn in record.conversation:
        speaker = turn.speaker.strip()
        text = turn.text.strip()
        turns_text.append(f"{speaker}: {text}")
    return " ".join(turns_text)

def preprocess_text(text: str) -> str:
    """
    Executes the cleaning pipeline:
    1. Lowercase
    2. Preserve scam entities
    3. Remove unnecessary punctuation
    4. Normalize whitespace and remove duplicate spaces
    5. Restore preserved entities
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Preserve scam entities
    placeholders = {}
    counter = 0
    temp_text = text

    for name, pattern in PRESERVE_PATTERNS:
        def repl(match):
            nonlocal counter
            val = match.group(0)
            placeholder = f"protectedentity{name.lower()}{counter}"
            placeholders[placeholder] = val
            counter += 1
            return placeholder
        temp_text = pattern.sub(repl, temp_text)

    # 3. Remove unnecessary punctuation (replace non-alphanumeric and non-whitespace characters with space)
    # This removes all punctuation except the letters/digits in our custom placeholders
    cleaned_text = re.sub(r"[^a-z0-9\s]", " ", temp_text)

    # 4. Normalize whitespace and collapse duplicate spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # 5. Restore preserved entities
    for placeholder, original_value in placeholders.items():
        cleaned_text = cleaned_text.replace(placeholder, original_value)

    return cleaned_text

def preprocess_pipeline(
    input_path: str,
    output_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    split_dataset: bool = True
) -> Dict[str, Any]:
    """
    Main function that runs the entire preprocessing pipeline:
    - Loads and validates raw JSON dataset
    - Runs text preprocessing and entity preservation
    - Performs stratified train/test split
    - Saves processed dataset as pickle
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_path}")

    logger.info(f"Loading dataset from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Input dataset must be a list of records.")

    valid_records: List[ConvertedScamRecord] = []
    invalid_count = 0

    logger.info("Validating dataset schema...")
    for idx, item in enumerate(raw_data):
        try:
            record = validate_record(item)
            valid_records.append(record)
        except Exception as e:
            invalid_count += 1
            if invalid_count <= 5:
                logger.warning(f"Validation failed for record at index {idx}: {e}")
    
    if invalid_count > 0:
        logger.warning(f"Total invalid records skipped: {invalid_count}")

    logger.info(f"Successfully validated {len(valid_records)} records.")

    # Process records
    texts = []
    labels = []
    risk_scores = []
    behavior_tags = []

    logger.info("Preprocessing conversation transcripts...")
    for record in valid_records:
        joined_text = join_turns(record)
        cleaned_text = preprocess_text(joined_text)
        
        texts.append(cleaned_text)
        labels.append(record.label)
        risk_scores.append(record.risk_score)
        behavior_tags.append(record.tags)

    # Check for empty output
    if not texts:
        raise ValueError("No valid records to output.")

    # Split dataset
    if split_dataset and test_size > 0.0:
        logger.info(f"Performing stratified train/test split (test_size={test_size}, random_state={random_state})...")
        from sklearn.model_selection import train_test_split
        import numpy as np

        indices = np.arange(len(texts))
        
        # Check label distribution to handle rare classes gracefully
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        rare_labels = [label for label, count in zip(unique_labels, label_counts) if count < 2]
        
        if rare_labels:
            logger.warning(f"Found rare labels with count < 2: {rare_labels}. Adjusting stratify target...")
            # For rare classes, we group them or split without stratify if they are too rare, or simply duplicate them for stratification.
            # To preserve stratification for the rest, we can use a fallback strategy where we split index-wise or adjust labels list.
            # Let's fallback to non-stratified split if any label has count < 2, or group rare labels.
            # Actually, to make stratification work, we need at least 2 samples per class. If not, we don't stratify.
            stratify_target = None
            logger.warning("Falling back to non-stratified split due to rare classes.")
        else:
            stratify_target = labels

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target
        )

        processed_data = {
            "train": {
                "texts": [texts[i] for i in train_idx],
                "labels": [labels[i] for i in train_idx],
                "risk_scores": [risk_scores[i] for i in train_idx],
                "behavior_tags": [behavior_tags[i] for i in train_idx]
            },
            "test": {
                "texts": [texts[i] for i in test_idx],
                "labels": [labels[i] for i in test_idx],
                "risk_scores": [risk_scores[i] for i in test_idx],
                "behavior_tags": [behavior_tags[i] for i in test_idx]
            }
        }
        logger.info(f"Train size: {len(train_idx)} | Test size: {len(test_idx)}")
    else:
        processed_data = {
            "train": {
                "texts": texts,
                "labels": labels,
                "risk_scores": risk_scores,
                "behavior_tags": behavior_tags
            },
            "test": {
                "texts": [],
                "labels": [],
                "risk_scores": [],
                "behavior_tags": []
            }
        }
        logger.info(f"Exported full dataset of size {len(texts)} without splitting.")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving preprocessed dataset to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)
    logger.info("Preprocessing complete.")

    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset preprocessing pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Path to input train_dataset.json")
    parser.add_argument("--output", type=str, default=None, help="Path to output preprocessed_dataset.pkl")
    parser.add_argument("--no-split", action="store_true", help="Disable train/test split")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine input path
    input_path = args.input
    if not input_path:
        # Default input path relative to project root
        input_path = project_root / "data" / "final" / "train_dataset.json"

    # Determine output path
    output_path = args.output
    if not output_path:
        # Default output path relative to project root
        output_path = project_root / "data" / "final" / "preprocessed_dataset.pkl"

    preprocess_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        test_size=0.0 if args.no_split else config.data.test_size,
        random_state=config.data.random_state,
        split_dataset=not args.no_split
    )
