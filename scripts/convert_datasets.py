import os
import re
import sys
import uuid
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# Add project root to sys.path to enable importing local src modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.modeling.reasoning import EvidenceExtractor
from src.data_pipeline.preprocessor import TextPreprocessor
from src.data_pipeline.nlp_pipeline import NLPPreprocessingPipeline
from src.data_pipeline.label_normalizer import LabelNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "data_conversion.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("convert_datasets")

# Ensure logs directory exists
(project_root / "logs").mkdir(parents=True, exist_ok=True)

# Define schemas for validation
class ConversationTurn(BaseModel):
    speaker: str = Field(..., description="The role or name of the talker.")
    text: str = Field(..., min_length=1, description="The transcript turn text.")

class ConvertedScamRecord(BaseModel):
    id: str = Field(..., description="UUIDv4 identifier.")
    language: str = Field("en", description="ISO 639-1 language code.")
    source: str = Field(..., description="Source dataset name.")
    conversation: List[ConversationTurn] = Field(..., description="Sequence of conversation turns.")
    label: str = Field(..., description="Unified classification label.")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk confidence score (0-100).")
    tags: Dict[str, bool] = Field(..., description="Binary flags indicating specific scam markers.")

class DatasetConverter:
    """Orchestrates dataset detection, cleaning, schema mapping, and validation."""

    def __init__(self):
        self.preprocessor = TextPreprocessor(lowercase=False)
        self.evidence_extractor = EvidenceExtractor()
        self.nlp_pipeline = NLPPreprocessingPipeline(lowercase=False)
        self.label_normalizer = LabelNormalizer()

    def detect_dataset_type(self, df: pd.DataFrame) -> str:
        """Heuristically detects dataset type based on column signatures."""
        cols = {c.lower() for c in df.columns}
        
        # 1. Fake Job Posting Dataset
        if {"title", "description", "requirements", "fraudulent"}.issubset(cols):
            return "fake_job_posting"
            
        # 2. Email Spam Dataset (various common formats)
        if {"text", "spam"}.issubset(cols) or {"message", "category"}.issubset(cols) or {"v1", "v2"}.issubset(cols):
            return "email_spam"
            
        # 3. Unified Scam Dataset
        if {"text", "label"}.issubset(cols) or {"conversation", "label"}.issubset(cols) or {"scammer_message", "is_scam"}.issubset(cols):
            return "unified_scam"
            
        return "unknown"

    def extract_scam_tags(self, text: str) -> Dict[str, bool]:
        """Runs the EvidenceExtractor reasoning engine to extract boolean indicator tags."""
        reasoning = self.evidence_extractor.extract_evidence(text)
        evidence = reasoning["evidence"]
        
        # Check for keep secret phrase explicitly
        keep_secret_keywords = ["secret", "don't tell", "do not share", "keep this to yourself", "confidential"]
        has_secret_key = any(k in text.lower() for k in keep_secret_keywords)

        return {
            "authority_impersonation": bool(evidence["authority_impersonation"]["detected"]),
            "urgency": bool(evidence["urgency"]["detected"]),
            "fear": bool(evidence["fear_tactics"]["detected"]),
            "money_request": bool(evidence["payment_request"]["detected"]),
            "otp_request": bool(evidence["otp_request"]["detected"]),
            "qr_request": bool(evidence["qr_code_request"]["detected"]),
            "bank_details_request": bool(evidence["bank_account_verification"]["detected"]),
            "remote_access_request": bool(evidence["remote_access_request"]["detected"]),
            "keep_secret": has_secret_key or bool(evidence["threat"]["detected"])
        }

    def convert_record(self, text: str, raw_label: Any, source_name: str) -> ConvertedScamRecord:
        """Converts raw text & labels into a validated ConvertedScamRecord instance."""
        # 1. Clean overall text first (needed for tags)
        cleaned_text = self.preprocessor.clean_text(text)
        
        # 2. Split speaker turns on raw text first to preserve tags
        speaker_turns = []
        
        # Check if text contains speaker boundaries (e.g. "[Speaker]:" or "Speaker:")
        if re.search(r"\[?[A-Za-z0-9\s_]+\]?\s*:\s*", text):
            raw_turns = self.nlp_pipeline.separate_speakers(text)
            for turn in raw_turns:
                cleaned_turn_text = self.preprocessor.clean_text(turn["text"])
                if cleaned_turn_text.strip():
                    speaker_turns.append(ConversationTurn(
                        speaker=turn["speaker"],
                        text=cleaned_turn_text
                    ))

        # Fallback if no speaker turns were extracted
        if not speaker_turns:
            speaker_turns.append(ConversationTurn(
                speaker="Sender",
                text=cleaned_text
            ))

        # Standardize labels and calculate baseline risk score using LabelNormalizer
        label_str = str(raw_label).strip()
        norm_label_enum = self.label_normalizer.normalize(cleaned_text, label_str)
        label = norm_label_enum.value
        risk_score = self.label_normalizer.get_risk_score(norm_label_enum)

        # Extract scam metadata tags using LabelNormalizer keyword matching
        tags = self.label_normalizer.detect_behavior_tags(cleaned_text)

        # Build and validate record
        record = ConvertedScamRecord(
            id=str(uuid.uuid4()),
            language="en",
            source=source_name,
            conversation=speaker_turns,
            label=label,
            risk_score=risk_score,
            tags=tags
        )
        return record

    def process_fake_job_posting(self, df: pd.DataFrame, source_name: str) -> List[Dict[str, Any]]:
        """Maps fake job posting dataset schema."""
        # Find exact case-insensitive column mappings
        col_map = {c.lower(): c for c in df.columns}
        desc_col = col_map["description"]
        req_col = col_map.get("requirements", desc_col)
        title_col = col_map.get("title", desc_col)
        fraud_col = col_map["fraudulent"]

        converted = []
        for _, row in df.iterrows():
            title = str(row[title_col]) if pd.notna(row[title_col]) else ""
            desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ""
            req = str(row[req_col]) if pd.notna(row[req_col]) else ""
            
            combined_text = f"Job Title: {title}. Description: {desc}. Requirements: {req}"
            if not combined_text.strip():
                continue
                
            try:
                record = self.convert_record(combined_text, row[fraud_col], source_name)
                converted.append(record.model_dump())
            except ValidationError as e:
                logger.debug(f"Row validation failed: {e}")
                
        return converted

    def process_email_spam(self, df: pd.DataFrame, source_name: str) -> List[Dict[str, Any]]:
        """Maps email/SMS spam dataset schema."""
        col_map = {c.lower(): c for c in df.columns}
        
        # Detect text column
        text_col = None
        for candidate in ["text", "message", "v2"]:
            if candidate in col_map:
                text_col = col_map[candidate]
                break
        
        # Detect label column
        label_col = None
        for candidate in ["spam", "category", "v1"]:
            if candidate in col_map:
                label_col = col_map[candidate]
                break

        if not text_col or not label_col:
            raise ValueError(f"Could not identify text/label columns in {source_name}")

        converted = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            if pd.isna(row[text_col]) or not text.strip():
                continue
                
            try:
                record = self.convert_record(text, row[label_col], source_name)
                converted.append(record.model_dump())
            except ValidationError as e:
                logger.debug(f"Row validation failed: {e}")
 
        return converted

    def process_unified_scam(self, df: pd.DataFrame, source_name: str) -> List[Dict[str, Any]]:
        """Maps unified scam datasets containing direct conversation/text and label fields."""
        col_map = {c.lower(): c for c in df.columns}
        
        # Check if this is the scammer_message / user_response / is_scam format
        if "scammer_message" in col_map and "is_scam" in col_map:
            text_col1 = col_map["scammer_message"]
            text_col2 = col_map.get("user_response")
            label_col = col_map["is_scam"]
            
            converted = []
            for _, row in df.iterrows():
                scammer_msg = str(row[text_col1]) if pd.notna(row[text_col1]) else ""
                user_msg = str(row[text_col2]) if text_col2 and pd.notna(row[text_col2]) else ""
                
                # Combine as dialog to be parsed by speaker turn separator
                combined_text = f"Scammer: {scammer_msg}"
                if user_msg.strip():
                    combined_text += f"\nUser: {user_msg}"
                    
                try:
                    record = self.convert_record(combined_text, row[label_col], source_name)
                    converted.append(record.model_dump())
                except ValidationError as e:
                    logger.debug(f"Row validation failed: {e}")
            return converted

        # Standard text/label or conversation/label formats
        text_col = col_map.get("text", col_map.get("conversation"))
        label_col = col_map.get("label")
        if not text_col or not label_col:
            raise ValueError(f"Could not identify text/label columns in {source_name}")

        converted = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            if pd.isna(row[text_col]) or not text.strip():
                continue
                
            try:
                record = self.convert_record(text, row[label_col], source_name)
                converted.append(record.model_dump())
            except ValidationError as e:
                logger.debug(f"Row validation failed: {e}")

        return converted

    def convert_file(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Processes a single file, dropping nulls and duplicates, validating, and writing JSON output."""
        source_name = file_path.stem
        logger.info(f"Processing dataset: '{file_path.name}'")

        # 1. Read CSV with fallback encoding if UTF-8 fails
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            logger.info(f"UTF-8 decode failed for {file_path.name}. Trying ISO-8859-1 encoding fallback.")
            try:
                df = pd.read_csv(file_path, encoding="ISO-8859-1")
            except Exception as e:
                logger.error(f"Failed to read file {file_path} with fallback encoding: {e}")
                return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return {"status": "error", "message": str(e)}

        total_rows = len(df)
        if total_rows == 0:
            logger.warning(f"File {file_path.name} is empty.")
            return {"status": "skipped", "message": "Empty file"}

        # 2. Detect Type
        dataset_type = self.detect_dataset_type(df)
        logger.info(f"Detected dataset type: '{dataset_type}'")
        
        if dataset_type == "unknown":
            logger.warning(f"Skipping file {file_path.name}: Unknown schema structure.")
            return {"status": "skipped", "message": "Unknown schema"}

        # 3. Drop Duplicates
        df = df.drop_duplicates()
        unique_rows = len(df)
        duplicates_removed = total_rows - unique_rows

        # 4. Map and Validate
        if dataset_type == "fake_job_posting":
            records = self.process_fake_job_posting(df, source_name)
        elif dataset_type == "email_spam":
            records = self.process_email_spam(df, source_name)
        elif dataset_type == "unified_scam":
            records = self.process_unified_scam(df, source_name)
        else:
            records = []

        valid_count = len(records)
        dropped_invalid = unique_rows - valid_count

        # 5. Save Converted File
        output_file = output_dir / f"{source_name}_converted.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved {valid_count} records to '{output_file.name}'")
        except Exception as e:
            logger.error(f"Failed to save JSON to {output_file}: {e}")
            return {"status": "error", "message": str(e)}

        return {
            "status": "success",
            "dataset_type": dataset_type,
            "total_records": total_rows,
            "duplicates_removed": duplicates_removed,
            "invalid_records_dropped": dropped_invalid,
            "valid_records_saved": valid_count
        }

def run_conversion() -> None:
    """Scans raw data directory recursively and runs the dataset converter pipeline."""
    external_dir = project_root / "data" / "external"
    processed_dir = project_root / "data" / "processed"

    # Create target directories
    external_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(external_dir.rglob("*.csv"))
    
    if not csv_files:
        logger.warning(
            f"No CSV datasets found inside data/external/ folder ('{external_dir.resolve()}'). "
            "Please copy dataset CSV files there to convert them."
        )
        return

    logger.info(f"Found {len(csv_files)} CSV file(s) to process recursively.")
    converter = DatasetConverter()
    stats = []

    for file_path in csv_files:
        res = converter.convert_file(file_path, processed_dir)
        if res["status"] == "success":
            stats.append({
                "Filename": file_path.name,
                "Type": res["dataset_type"],
                "Total Rows": res["total_records"],
                "Duplicates Removed": res["duplicates_removed"],
                "Invalid Dropped": res["invalid_records_dropped"],
                "Saved Count": res["valid_records_saved"]
            })

    # Output Statistics Summary
    if stats:
        stats_df = pd.DataFrame(stats)
        print("\n================== CONVERSION STATISTICS SUMMARY ==================")
        print(stats_df.to_string(index=False))
        print("===================================================================\n")
        logger.info("Conversion summary printed to stdout.")

if __name__ == "__main__":
    run_conversion()
