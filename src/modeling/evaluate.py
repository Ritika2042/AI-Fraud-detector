import json
from pathlib import Path
from typing import List, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_predictions(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """
    Computes performance metrics comparing ground truth labels and predictions.
    Returns a structured dictionary of metrics including class-specific statistics
    and the confusion matrix.
    """
    # Force element conversion to strings
    y_true_str = [str(y) for y in y_true]
    y_pred_str = [str(y) for y in y_pred]
    labels_str = sorted(list([str(l) for l in labels]))

    # Calculate overall accuracy
    accuracy = float(accuracy_score(y_true_str, y_pred_str))

    # Generate the classification report dictionary
    report_dict = classification_report(
        y_true_str,
        y_pred_str,
        labels=labels_str,
        output_dict=True,
        zero_division=0
    )

    # Generate confusion matrix
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels_str)

    # Assemble metrics structure
    metrics = {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": float(report_dict["macro avg"]["precision"]),
            "recall": float(report_dict["macro avg"]["recall"]),
            "f1-score": float(report_dict["macro avg"]["f1-score"]),
            "support": int(report_dict["macro avg"]["support"])
        },
        "weighted_avg": {
            "precision": float(report_dict["weighted avg"]["precision"]),
            "recall": float(report_dict["weighted avg"]["recall"]),
            "f1-score": float(report_dict["weighted avg"]["f1-score"]),
            "support": int(report_dict["weighted avg"]["support"])
        },
        "class_metrics": {},
        "confusion_matrix": cm.tolist(),
        "labels": labels_str
    }

    # Extract per-class precision, recall, f1-score
    for label in labels_str:
        if label in report_dict:
            metrics["class_metrics"][label] = {
                "precision": float(report_dict[label]["precision"]),
                "recall": float(report_dict[label]["recall"]),
                "f1-score": float(report_dict[label]["f1-score"]),
                "support": int(report_dict[label]["support"])
            }

    return metrics

def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
    """Saves the evaluation metrics dictionary to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
