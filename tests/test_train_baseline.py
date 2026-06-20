import os
import sys
import tempfile
import json
import pickle
import pytest
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.modeling.train_baseline import (
    load_preprocessed_dataset,
    train_and_tune_model,
    evaluate_model,
    get_feature_importance,
    get_most_confused_classes,
    generate_baseline_report
)

@pytest.fixture
def dummy_dataset():
    # 15 dummy training records across 3 classes
    X_train = [
        "please pay rs. 5000 using upi to claim your free reward immediately",
        "urgent bank kyc verification needed to unblock your account",
        "congratulations you won a personal loan from hdfc bank click here",
        "this is safe and normal talk about weather and family",
        "safe message no scam detected in this conversation turn",
        "please share the otp code to complete the verification",
        "digital arrest warning you are under cbi investigation cooperation required",
        "transfer money to axis bank account now or face jail",
        "safe conversation about project details and schedule",
        "hello grandfather is everything okay at home i will call you",
        "please transfer processing fee to the agent's account",
        "your sbi credit card has been frozen call support now",
        "safe email reminder about meeting tomorrow morning",
        "romance and chat safe hello how is your day going my dear",
        "loan approved click link to pay registration fee"
    ]
    y_train = [
        "OTP Scam", "Bank KYC Scam", "Loan Scam", "Safe", "Safe",
        "OTP Scam", "Digital Arrest Scam", "Bank KYC Scam", "Safe", "Safe",
        "Loan Scam", "Bank KYC Scam", "Safe", "Safe", "Loan Scam"
    ]
    
    # 5 dummy test records
    X_test = [
        "please pay the fee immediately using upi or bank transfer",
        "safe message no urgency here at all",
        "read out the otp code now",
        "your account is blocked do kyc verification",
        "hello are we going to have dinner tonight"
    ]
    y_test = [
        "Loan Scam", "Safe", "OTP Scam", "Bank KYC Scam", "Safe"
    ]
    
    return {
        "train": {
            "texts": X_train,
            "labels": y_train,
            "risk_scores": [80.0 if y != "Safe" else 0.0 for y in y_train],
            "behavior_tags": [{"urgency": True} if y != "Safe" else {"urgency": False} for y in y_train]
        },
        "test": {
            "texts": X_test,
            "labels": y_test,
            "risk_scores": [80.0 if y != "Safe" else 0.0 for y in y_test],
            "behavior_tags": [{"urgency": True} if y != "Safe" else {"urgency": False} for y in y_test]
        }
    }

def test_load_preprocessed_dataset(dummy_dataset):
    with tempfile.TemporaryDirectory() as temp_dir:
        pkl_path = Path(temp_dir) / "preprocessed_dataset.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(dummy_dataset, f)
            
        train, test = load_preprocessed_dataset(pkl_path)
        assert len(train["texts"]) == 15
        assert len(test["texts"]) == 5
        assert train["labels"][0] == "OTP Scam"

def test_train_and_tune_model(dummy_dataset):
    X_train = dummy_dataset["train"]["texts"]
    y_train = dummy_dataset["train"]["labels"]
    
    # Fast parameter grid
    param_grid = {
        "tfidf__ngram_range": [(1, 1)],
        "tfidf__min_df": [1],
        "clf__C": [1.0]
    }
    
    clf = LogisticRegression(random_state=42)
    best_pipeline, elapsed = train_and_tune_model("logistic", clf, param_grid, X_train, y_train)
    
    assert isinstance(best_pipeline, Pipeline)
    assert elapsed > 0
    assert hasattr(best_pipeline.named_steps["clf"], "coef_")

def test_evaluate_model(dummy_dataset):
    X_train = dummy_dataset["train"]["texts"]
    y_train = dummy_dataset["train"]["labels"]
    X_test = dummy_dataset["test"]["texts"]
    y_test = dummy_dataset["test"]["labels"]
    
    classes = sorted(list(set(y_train)))
    
    # Train a quick pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1)),
        ("clf", LogisticRegression(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    metrics, latency = evaluate_model(pipeline, X_test, y_test, classes)
    
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "class_metrics" in metrics
    assert len(metrics["confusion_matrix"]) == len(classes)
    assert latency > 0

def test_get_feature_importance(dummy_dataset):
    X_train = dummy_dataset["train"]["texts"]
    y_train = dummy_dataset["train"]["labels"]
    classes = sorted(list(set(y_train)))
    
    # 1. Test Logistic Regression
    pipeline_lr = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1)),
        ("clf", LogisticRegression(random_state=42))
    ])
    pipeline_lr.fit(X_train, y_train)
    feat_lr = get_feature_importance(pipeline_lr, classes, top_n=2)
    for c in classes:
        assert c in feat_lr
        assert len(feat_lr[c]) <= 2

    # 2. Test SVM
    pipeline_svm = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1)),
        ("clf", LinearSVC(dual=False, random_state=42))
    ])
    pipeline_svm.fit(X_train, y_train)
    feat_svm = get_feature_importance(pipeline_svm, classes, top_n=2)
    for c in classes:
        assert c in feat_svm

    # 3. Test Naive Bayes
    pipeline_nb = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1)),
        ("clf", MultinomialNB())
    ])
    pipeline_nb.fit(X_train, y_train)
    feat_nb = get_feature_importance(pipeline_nb, classes, top_n=2)
    for c in classes:
        assert c in feat_nb

    # 4. Test Random Forest
    pipeline_rf = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1)),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    pipeline_rf.fit(X_train, y_train)
    feat_rf = get_feature_importance(pipeline_rf, classes, top_n=2)
    assert "overall" in feat_rf
    assert len(feat_rf["overall"]) <= 4

def test_get_most_confused_classes():
    cm = [
        [10, 2, 0],
        [1, 15, 4],
        [0, 3, 8]
    ]
    classes = ["Safe", "OTP Scam", "Loan Scam"]
    confusions = get_most_confused_classes(cm, classes, top_n=2)
    
    assert len(confusions) == 2
    # The highest off-diagonal is cm[1][2] = 4 (OTP Scam -> Loan Scam)
    assert confusions[0][0] == "OTP Scam"
    assert confusions[0][1] == "Loan Scam"
    assert confusions[0][2] == 4

def test_generate_baseline_report():
    results = {
        "logistic": {
            "best_params": {"tfidf__ngram_range": "(1, 1)", "clf__C": "1.0"},
            "metrics": {
                "accuracy": 0.8,
                "precision_macro": 0.8,
                "recall_macro": 0.8,
                "f1_macro": 0.8,
                "f1_weighted": 0.8,
                "confusion_matrix": [[5, 0], [2, 3]],
                "classes": ["Safe", "Scam"],
                "class_metrics": {
                    "Safe": {"precision": 0.71, "recall": 1.0, "f1": 0.83, "support": 5},
                    "Scam": {"precision": 1.0, "recall": 0.6, "f1": 0.75, "support": 5}
                }
            },
            "train_time": 0.5,
            "pred_time": 0.05,
            "test_size": 10,
            "feature_importance": {
                "Safe": [{"feature": "hello", "importance": 0.5}],
                "Scam": [{"feature": "pay", "importance": 0.8}]
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = Path(temp_dir) / "baseline_report.md"
        generate_baseline_report(results, "logistic", report_path)
        
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Baseline Scam Detection Models Evaluation Report" in content
        assert "Logistic" in content
        assert "0.8000" in content
