import os
import sys
import tempfile
import json
import pickle
import pytest
from pathlib import Path
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.modeling.train_transformer_experiment import (
    TransformerTextDataset,
    compute_metrics,
    try_load_model_and_tokenizer,
    write_reports
)

@pytest.fixture
def dummy_tokenized_data():
    encodings = {
        "input_ids": [[101, 1000, 102], [101, 2000, 102], [101, 3000, 102]],
        "attention_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    }
    labels = [0, 1, 0]
    return encodings, labels

def test_transformer_text_dataset(dummy_tokenized_data):
    encodings, labels = dummy_tokenized_data
    dataset = TransformerTextDataset(encodings, labels)
    
    assert len(dataset) == 3
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"].item() == 0

def test_compute_metrics():
    # Mock eval pred
    logits = np.array([
        [2.0, -1.0],
        [-0.5, 1.5],
        [3.0, 0.0]
    ]) # Argmax predictions: [0, 1, 0]
    labels = np.array([0, 1, 1]) # Ground truth
    
    # Accuracy: 2 out of 3 = 0.6667
    eval_pred = (logits, labels)
    res = compute_metrics(eval_pred)
    
    assert "accuracy" in res
    assert "f1_macro" in res
    assert abs(res["accuracy"] - 0.6667) < 0.001

def test_try_load_model_and_tokenizer_fallback():
    # If the first model name in the priority list is fake/invalid, it should fail
    # and fallback to a valid tiny model name.
    model_priority = [
        "microsoft/this-is-a-fake-unfindable-model-backbone",
        "hf-internal-testing/tiny-random-BertForSequenceClassification"
    ]
    label_to_id = {"Safe": 0, "Scam": 1}
    id_to_label = {0: "Safe", 1: "Scam"}
    
    model_name, tokenizer, model = try_load_model_and_tokenizer(
        model_priority=model_priority,
        num_labels=2,
        label_to_id=label_to_id,
        id_to_label=id_to_label
    )
    
    assert model_name == "hf-internal-testing/tiny-random-BertForSequenceClassification"
    assert tokenizer is not None
    assert model is not None

def test_write_reports():
    metrics = {
        "accuracy": 0.90,
        "precision_macro": 0.88,
        "recall_macro": 0.89,
        "f1_macro": 0.885,
        "f1_weighted": 0.90,
        "confusion_matrix": [[5, 1], [0, 4]],
        "class_metrics": {
            "Safe": {"precision": 1.0, "recall": 0.83, "f1": 0.91, "support": 6},
            "Scam": {"precision": 0.8, "recall": 1.0, "f1": 0.89, "support": 4}
        }
    }
    
    svm_metrics = {
        "accuracy": 0.85,
        "precision_macro": 0.82,
        "recall_macro": 0.83,
        "f1_macro": 0.825,
        "f1_weighted": 0.85
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "transformer_report.md"
        comparison_path = Path(temp_dir) / "model_comparison.md"
        
        write_reports(
            selected_model_name="tiny-bert",
            device="cpu",
            epochs=1,
            batch_size=8,
            metrics=metrics,
            classes=["Safe", "Scam"],
            svm_metrics=svm_metrics,
            report_path=report_path,
            comparison_path=comparison_path,
            training_time=12.5,
            pred_latency_ms=1.5
        )
        
        assert report_path.exists()
        assert comparison_path.exists()
        
        report_content = report_path.read_text(encoding="utf-8")
        comparison_content = comparison_path.read_text(encoding="utf-8")
        
        assert "tiny-bert" in report_content
        assert "0.8850" in report_content
        assert "SVM Baseline" in comparison_content
        assert "0.8250" in comparison_content
