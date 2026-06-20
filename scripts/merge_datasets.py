import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import numpy as np

# Add project root to sys.path to enable local imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.convert_datasets import ConvertedScamRecord, ConversationTurn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "dataset_merger.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("merge_datasets")

# Ensure logs directory exists
(project_root / "logs").mkdir(parents=True, exist_ok=True)

def get_conversation_key(conversation: List[ConversationTurn]) -> Tuple[Tuple[str, str], ...]:
    """Generates a unique hashable key for conversation turns to identify duplicates."""
    return tuple((turn.speaker.strip(), turn.text.strip().lower()) for turn in conversation)

def merge_and_deduplicate(processed_dir: Path, synthetic_dir: Path) -> List[ConvertedScamRecord]:
    """Reads all JSON files in target directories, validates schema, and removes duplicates."""
    seen_ids: Set[str] = set()
    seen_conversations: Set[Tuple[Tuple[str, str], ...]] = set()
    merged_records: List[ConvertedScamRecord] = []
    
    # Collect all JSON files
    json_files: List[Path] = []
    
    for directory in [processed_dir, synthetic_dir]:
        if directory.exists():
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".json"):
                        json_files.append(Path(root) / file)
                        
    logger.info(f"Scanning {len(json_files)} JSON files across directories...")
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.debug(f"Skipping {file_path.name}: Root element is not a JSON list.")
                continue
                
            file_records_count = 0
            for item in data:
                try:
                    # Validate schema
                    record = ConvertedScamRecord(**item)
                    file_records_count += 1
                    
                    # Deduplicate by ID
                    if record.id in seen_ids:
                        logger.debug(f"Skipping duplicate ID: {record.id}")
                        continue
                        
                    # Deduplicate by conversation content
                    conv_key = get_conversation_key(record.conversation)
                    if conv_key in seen_conversations:
                        logger.debug(f"Skipping duplicate conversation content in record {record.id}")
                        continue
                        
                    seen_ids.add(record.id)
                    seen_conversations.add(conv_key)
                    merged_records.append(record)
                except Exception:
                    # Not a valid ConvertedScamRecord, skip silently (e.g. graphs.json)
                    continue
            
            logger.info(f"Loaded {file_records_count} valid records from {file_path.name}.")
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            
    logger.info(f"Merge complete. Total unique records: {len(merged_records)} (duplicates removed).")
    return merged_records

def compute_statistics(records: List[ConvertedScamRecord]) -> Dict[str, Any]:
    """Computes advanced dataset statistics and distributions."""
    if not records:
        return {}
        
    total_records = len(records)
    
    # Label and source distributions
    label_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    risk_scores: List[float] = []
    total_turns_list: List[int] = []
    word_counts_list: List[int] = []
    
    for r in records:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1
        source_counts[r.source] = source_counts.get(r.source, 0) + 1
        risk_scores.append(r.risk_score)
        
        # Turn count
        turns_count = len(r.conversation)
        total_turns_list.append(turns_count)
        
        # Word count
        word_count = sum(len(turn.text.split()) for turn in r.conversation)
        word_counts_list.append(word_count)
        
    # Risk scores metrics
    risk_min = float(np.min(risk_scores))
    risk_max = float(np.max(risk_scores))
    risk_mean = float(np.mean(risk_scores))
    risk_median = float(np.median(risk_scores))
    risk_std = float(np.std(risk_scores))
    
    # Class imbalance ratio calculation
    # majority / minority count
    label_sorted = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    majority_class, majority_count = label_sorted[0]
    minority_class, minority_count = label_sorted[-1]
    class_imbalance_ratio = majority_count / minority_count if minority_count > 0 else float("inf")
    
    # Ratios relative to majority class
    class_ratios = {label: count / majority_count for label, count in label_counts.items()}
    
    # Averages
    avg_turns = float(np.mean(total_turns_list))
    avg_words = float(np.mean(word_counts_list))
    
    return {
        "total_records": total_records,
        "label_counts": label_counts,
        "source_counts": source_counts,
        "risk_stats": {
            "min": risk_min,
            "max": risk_max,
            "mean": risk_mean,
            "median": risk_median,
            "std": risk_std
        },
        "imbalance": {
            "majority_class": majority_class,
            "majority_count": majority_count,
            "minority_class": minority_class,
            "minority_count": minority_count,
            "ratio": class_imbalance_ratio,
            "class_ratios": class_ratios
        },
        "avg_turns": avg_turns,
        "avg_words": avg_words,
        "top_labels": label_sorted[:20]
    }

def export_statistics_report(stats: Dict[str, Any], report_path: Path):
    """Exports dataset statistics to a markdown report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Consolidated Train Dataset Statistics Report\n\n")
        rf.write("This report provides a comprehensive summary and statistical analysis of the consolidated train dataset.\n\n")
        
        rf.write("## 1. Summary Statistics\n")
        rf.write(f"*   **Total Unique Records:** `{stats['total_records']}`\n")
        rf.write(f"*   **Average Conversation Turns:** `{stats['avg_turns']:.2f}` turns\n")
        rf.write(f"*   **Average Words per Conversation:** `{stats['avg_words']:.2f}` words\n")
        rf.write(f"*   **Class Imbalance Ratio (Majority:Minority):** `{stats['imbalance']['ratio']:.2f}:1` ")
        rf.write(f"(Majority: `{stats['imbalance']['majority_class']}` ({stats['imbalance']['majority_count']}), ")
        rf.write(f"Minority: `{stats['imbalance']['minority_class']}` ({stats['imbalance']['minority_count']}))\n\n")
        
        rf.write("## 2. Risk Score Distribution\n")
        rf.write("| Metric | Value |\n")
        rf.write("| :--- | :---: |\n")
        rf.write(f"| Minimum Risk Score | {stats['risk_stats']['min']:.2f} |\n")
        rf.write(f"| Maximum Risk Score | {stats['risk_stats']['max']:.2f} |\n")
        rf.write(f"| Mean Risk Score | {stats['risk_stats']['mean']:.2f} |\n")
        rf.write(f"| Median Risk Score | {stats['risk_stats']['median']:.2f} |\n")
        rf.write(f"| Standard Deviation | {stats['risk_stats']['std']:.2f} |\n\n")
        
        rf.write("## 3. Label Distribution & Class Imbalance Ratios\n")
        rf.write("| Scam Category Label | Count | Percentage | Ratio to Majority Class |\n")
        rf.write("| :--- | :---: | :---: | :---: |\n")
        for label, count in sorted(stats['label_counts'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / stats['total_records']) * 100.0
            ratio = stats['imbalance']['class_ratios'][label]
            rf.write(f"| {label} | {count} | {pct:.2f}% | {ratio:.4f} |\n")
        rf.write("\n")
        
        rf.write("## 4. Source Distribution\n")
        rf.write("| Dataset Source | Count | Percentage |\n")
        rf.write("| :--- | :---: | :---: |\n")
        for src, count in sorted(stats['source_counts'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / stats['total_records']) * 100.0
            rf.write(f"| {src} | {count} | {pct:.2f}% |\n")
        rf.write("\n")
        
        rf.write("## 5. Top 20 Most Frequent Scam Labels\n")
        rf.write("| Rank | Scam Category Label | Record Count |\n")
        rf.write("| :---: | :--- | :---: |\n")
        for idx, (label, count) in enumerate(stats['top_labels']):
            rf.write(f"| {idx + 1} | {label} | {count} |\n")
            
    logger.info(f"Statistics report successfully saved to {report_path}.")

def main():
    processed_dir = project_root / "data" / "processed"
    synthetic_train_dir = project_root / "data" / "synthetic" / "train_templates"
    output_dir = project_root / "data" / "final"
    report_file = project_root / "docs" / "final_dataset_statistics.md"
    
    # 1. Merge and deduplicate
    merged_records = merge_and_deduplicate(processed_dir, synthetic_train_dir)
    
    if not merged_records:
        logger.error("No valid records found to merge. Exiting.")
        return
        
    # 2. Compute statistics
    stats = compute_statistics(merged_records)
    
    # 3. Print summary to console
    print("\n================ DATASET MERGER STATISTICS ================")
    print(f"Total Unique Records: {stats['total_records']}")
    print(f"Average Conversation Turns: {stats['avg_turns']:.2f}")
    print(f"Average Words per Conversation: {stats['avg_words']:.2f}")
    print(f"Class Imbalance Ratio (Max/Min): {stats['imbalance']['ratio']:.2f}")
    print("\nSource Distribution:")
    for src, count in sorted(stats['source_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {src}: {count} ({count/stats['total_records']*100:.2f}%)")
    print("\nTop Labels (up to 20):")
    for idx, (label, count) in enumerate(stats['top_labels']):
        print(f"  {idx+1:2d}. {label}: {count} ({count/stats['total_records']*100:.2f}%)")
    print("===========================================================\n")
    
    # 4. Export report to docs/
    export_statistics_report(stats, report_file)
    
    # 5. Save final train dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_dataset.json"
    
    serialized_records = []
    for r in merged_records:
        if hasattr(r, "model_dump"):
            serialized_records.append(r.model_dump())
        else:
            serialized_records.append(r.dict())
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized_records, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Consolidated dataset saved to {output_path}.")

if __name__ == "__main__":
    main()
