import os
import sys
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add workspace to sys.path
workspace_root = Path(r"E:\Hackathons_Projects\AI Fraud detector")
sys.path.append(str(workspace_root))

from src.config import load_config
from src.data_pipeline.preprocessor import TextPreprocessor

def get_jaccard_sim(str1: str, str2: str) -> float:
    """Computes word-level Jaccard similarity between two strings."""
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    if not words1 and not words2:
        return 1.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return float(len(intersection)) / len(union)

def run_leakage_audit(config_path: str = None) -> None:
    # 1. Load config
    config = load_config(config_path)
    raw_path = workspace_root / config.data.raw_data_path
    
    print("==========================================================")
    print("                DATA LEAKAGE AUDIT RUNNER                 ")
    print("==========================================================\n")
    
    if not raw_path.exists():
        print(f"Error: Raw data not found at '{raw_path}'")
        return

    # 2. Read raw data
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Total raw rows: {len(df)}")
    
    # 3. Preprocess texts using identical settings as train.py
    preprocessor = TextPreprocessor(lowercase=config.model.tfidf.lowercase)
    df_clean = preprocessor.preprocess_df(df.copy(), text_column="text")
    
    texts = df_clean["text"].tolist()
    labels = df_clean["label"].tolist()
    
    # 4. Simulate train/test split
    test_size = config.data.test_size
    random_state = config.data.random_state
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    print(f"Simulating train/test split (Test size: {test_size}, Random State: {random_state})")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Check 1: ID Duplication
    # Since raw data might not have IDs, we check if there are duplicate indexes or ID columns
    id_duplicates = 0
    if "id" in df.columns:
        id_duplicates = df["id"].duplicated().sum()
    elif "id" in df.index.names:
        id_duplicates = df.index.duplicated().sum()

    # Check 2: Exact duplicates between train and test
    exact_duplicates = 0
    exact_dupe_details = []
    
    train_set = set(X_train)
    for idx, test_text in enumerate(X_test):
        if test_text in train_set:
            exact_duplicates += 1
            exact_dupe_details.append((test_text, y_test[idx]))

    # Check 3: Near-duplicates (Jaccard similarity > 0.8) between train and test
    near_duplicates = 0
    near_dupe_details = []
    
    # To be fast, check only test samples that are not exact duplicates
    for idx, test_text in enumerate(X_test):
        if test_text in train_set:
            continue
        max_sim = 0.0
        best_match = ""
        for train_text in X_train:
            sim = get_jaccard_sim(test_text, train_text)
            if sim > max_sim:
                max_sim = sim
                best_match = train_text
                
        if max_sim >= 0.8:
            near_duplicates += 1
            near_dupe_details.append((test_text, best_match, max_sim, y_test[idx]))

    # Check 4: Vocabulary Overlap
    train_vocab = set(" ".join(X_train).lower().split())
    test_vocab = set(" ".join(X_test).lower().split())
    vocab_overlap = len(train_vocab.intersection(test_vocab))
    vocab_jaccard = float(vocab_overlap) / len(train_vocab.union(test_vocab)) if train_vocab or test_vocab else 1.0

    # Check 5: Augmentation Split Order check
    # In this pipeline, synthetic generation happens globally (via generate_synthetic_data.py)
    # and then the split happens on the raw_data_path.
    # Therefore, the split occurred AFTER synthetic templates were augmented/duplicated.
    split_before_augmentation = False
    
    # Calculate leakage metrics
    total_leakage_cases = exact_duplicates + near_duplicates
    leakage_percentage = (total_leakage_cases / len(X_test)) * 100
    
    print("\n--- AUDIT RESULTS ---")
    print(f"1. Exact duplicates between Train/Test: {exact_duplicates} ({ (exact_duplicates/len(X_test))*100:.2f}%)")
    print(f"2. Near-duplicates (Jaccard > 0.8)    : {near_duplicates} ({ (near_duplicates/len(X_test))*100:.2f}%)")
    print(f"3. Duplicated IDs                     : {id_duplicates}")
    print(f"4. Vocabulary overlap size            : {vocab_overlap} words")
    print(f"5. Vocabulary Jaccard similarity      : {vocab_jaccard:.4f}")
    print(f"6. Total Leakage Risk Percentage      : {leakage_percentage:.2f}%")
    print(f"7. Split before augmentation occurred : {split_before_augmentation}")

    # Generate recommendations
    has_leakage = leakage_percentage > 0.0 or not split_before_augmentation
    recommendations = []
    if has_leakage:
        recommendations.append("*   **Split Before Augmentation:** Redesign the dataset creation pipeline. Do not split the generated CSV. Instead, split the seed templates or raw prompts *first* into training templates and validation templates. Generate synthetic data independently for train and test sets using these separate templates.")
        recommendations.append("*   **Deduplicate Before Splitting:** Run a text-level deduplication filter on the raw text before splitting, to remove exact duplicate strings.")
        recommendations.append("*   **Strict Similarity Holdout Filter:** When splitting, calculate the Jaccard similarity between test records and train records. Purge any test records that have a Jaccard similarity > 0.6 with *any* training record of the same label.")

    # 10. Generate docs/data_leakage_report.md
    docs_dir = workspace_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_file = docs_dir / "data_leakage_report.md"
    
    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("# Data Leakage Audit & Verification Report\n\n")
        rf.write("This report validates the integrity of the train/test datasets and split methodologies to identify sources of data contamination and explain the 100% evaluation accuracy.\n\n")
        
        rf.write("## 1. Executive Summary\n")
        status_str = "CRITICAL LEAKAGE DETECTED" if has_leakage else "NO SIGNIFICANT LEAKAGE DETECTED"
        rf.write(f"*   **Audit Status:** `{status_str}`\n")
        rf.write(f"*   **Possible Leakage Percentage:** `{leakage_percentage:.2f}%`\n")
        rf.write(f"*   **Split Order Validation:** `Split occurred AFTER synthetic augmentation` (Flagged: **High Risk**)\n\n")
        
        rf.write("## 2. Detailed Audit Metrics\n")
        rf.write("| Check Description | Metric Value | Risk Level |\n")
        rf.write("| :--- | :---: | :---: |\n")
        rf.write(f"| Exact duplicate conversations between splits | {exact_duplicates} ({ (exact_duplicates/len(X_test))*100:.2f}%) | {'High' if exact_duplicates > 0 else 'Low'} |\n")
        rf.write(f"| Near-duplicate conversations (Jaccard > 0.8) | {near_duplicates} ({ (near_duplicates/len(X_test))*100:.2f}%) | {'High' if near_duplicates > 0 else 'Low'} |\n")
        rf.write(f"| Duplicated IDs found in dataset | {id_duplicates} | {'Medium' if id_duplicates > 0 else 'Low'} |\n")
        rf.write(f"| Vocabulary overlap size | {vocab_overlap} words | Informational |\n")
        rf.write(f"| Vocabulary Jaccard similarity index | {vocab_jaccard:.4f} | Informational |\n")
        rf.write(f"| Train/Test split occurred before augmentation | {split_before_augmentation} | {'Critical' if not split_before_augmentation else 'Low'} |\n\n")
        
        rf.write("## 3. Findings & Technical Analysis\n")
        rf.write("### Template Contamination\n")
        rf.write("The primary source of data leakage is **Template Contamination**. Because the synthetic dataset was generated by formatting a small pool of static templates repeatedly, many records share identical sentence structures and differ only in minor variable details (names, amounts, banks).\n\n")
        rf.write("When a random `train_test_split` is executed, these near-duplicate templates are distributed into both training and test sets. The model simply memorizes the static template structure (which maps 1-to-1 to the labels), resulting in an artificially inflated 100% evaluation accuracy.\n\n")
        
        if exact_dupe_details:
            rf.write("### Sample Exact Duplicates in Test Set\n")
            for text, lbl in exact_dupe_details[:3]:
                rf.write(f"*   **Label:** `{lbl}` | **Text:** *\"{text[:100]}...\"*\n")
            rf.write("\n")
            
        if near_dupe_details:
            rf.write("### Sample Near-Duplicates in Test Set\n")
            for test_t, train_t, sim, lbl in near_dupe_details[:3]:
                rf.write(f"*   **Label:** `{lbl}` (Similarity: `{sim:.4f}`)\n")
                rf.write(f"    *   *Test:* \"{test_t[:100]}...\"\n")
                rf.write(f"    *   *Train:* \"{train_t[:100]}...\"\n")
            rf.write("\n")

        rf.write("## 4. Recommendations & Fixes\n")
        if recommendations:
            for rec in recommendations:
                rf.write(f"{rec}\n")
        else:
            rf.write("No recommendations required. Dataset split protocol is secure.\n")
            
    print(f"\nVerification report generated successfully at: {report_file.resolve()}")

if __name__ == "__main__":
    run_leakage_audit()
