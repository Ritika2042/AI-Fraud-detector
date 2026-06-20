import os
import sys
import time
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add project root to sys.path to enable local imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import load_config
from src.logger import get_logger

logger = get_logger()

# Import ML requirements
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_preprocessed_dataset(pkl_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads the preprocessed dataset splits from pickle."""
    if not pkl_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at {pkl_path}")
    logger.info(f"Loading preprocessed dataset from {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["train"], data["test"]

def train_and_tune_model(
    model_name: str,
    clf,
    param_grid: Dict[str, List[Any]],
    X_train: List[str],
    y_train: List[str]
) -> Tuple[Pipeline, float]:
    """Sets up a pipeline, runs grid search tuning, and returns the best estimator and elapsed time."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", clf)
    ])

    logger.info(f"Running Grid Search for {model_name}...")
    start_time = time.time()
    
    # Use GridSearchCV with 3-fold cross validation and all CPU cores
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Finished tuning {model_name} in {elapsed_time:.2f}s. Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, elapsed_time

def evaluate_model(
    model: Pipeline,
    X_test: List[str],
    y_test: List[str],
    classes: List[str]
) -> Tuple[Dict[str, Any], float]:
    """Evaluates the model on test split and computes predictions, metrics, and latency."""
    start_time = time.time()
    predictions = model.predict(X_test)
    elapsed_prediction_time = time.time() - start_time
    
    # Standard sklearn metrics
    accuracy = float(accuracy_score(y_test, predictions))
    
    report_dict = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(y_test, predictions, labels=classes)
    
    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(report_dict["macro avg"]["precision"]),
        "recall_macro": float(report_dict["macro avg"]["recall"]),
        "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "class_metrics": {}
    }
    
    for cls in classes:
        if cls in report_dict:
            metrics["class_metrics"][cls] = {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }
        else:
            metrics["class_metrics"][cls] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0
            }
            
    return metrics, elapsed_prediction_time

def get_feature_importance(model: Pipeline, classes: List[str], top_n: int = 10) -> Dict[str, Any]:
    """Extracts top features for each class based on coefficients/probabilities/importances."""
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    
    importance_info = {}
    
    if isinstance(clf, (LogisticRegression, LinearSVC)):
        # Binary vs Multiclass coefficients
        coef = clf.coef_
        is_binary = coef.shape[0] == 1
        
        if is_binary:
            # For binary classification, there is only one row of coefficients for the positive class (classes[1])
            # Classes[0] has negative coefficients
            for class_idx, class_name in enumerate(classes):
                sorted_idx = np.argsort(coef[0])
                if class_idx == 1:
                    # Positive class
                    top_idx = sorted_idx[-top_n:][::-1]
                else:
                    # Negative class
                    top_idx = sorted_idx[:top_n]
                
                importance_info[class_name] = [
                    {"feature": str(feature_names[idx]), "importance": float(coef[0, idx])}
                    for idx in top_idx
                ]
        else:
            # Multiclass
            for class_idx, class_name in enumerate(classes):
                # Ensure the class index is within the coef_ bounds
                # Sometimes classifier.classes_ has a different order than evaluate classes
                clf_classes = list(clf.classes_)
                if class_name in clf_classes:
                    clf_class_idx = clf_classes.index(class_name)
                    sorted_idx = np.argsort(coef[clf_class_idx])
                    top_idx = sorted_idx[-top_n:][::-1]
                    importance_info[class_name] = [
                        {"feature": str(feature_names[idx]), "importance": float(coef[clf_class_idx, idx])}
                        for idx in top_idx
                    ]
                else:
                    importance_info[class_name] = []
                    
    elif isinstance(clf, MultinomialNB):
        log_prob = clf.feature_log_prob_
        for class_idx, class_name in enumerate(classes):
            clf_classes = list(clf.classes_)
            if class_name in clf_classes:
                clf_class_idx = clf_classes.index(class_name)
                sorted_idx = np.argsort(log_prob[clf_class_idx])
                top_idx = sorted_idx[-top_n:][::-1]
                importance_info[class_name] = [
                    {"feature": str(feature_names[idx]), "importance": float(log_prob[clf_class_idx, idx])}
                    for idx in top_idx
                ]
            else:
                importance_info[class_name] = []
                
    elif isinstance(clf, RandomForestClassifier):
        importances = clf.feature_importances_
        sorted_idx = np.argsort(importances)
        top_idx = sorted_idx[-top_n * 2:][::-1] # Get top 20 overall features
        importance_info["overall"] = [
            {"feature": str(feature_names[idx]), "importance": float(importances[idx])}
            for idx in top_idx
        ]
        
    return importance_info

def get_most_confused_classes(cm: List[List[int]], classes: List[str], top_n: int = 5) -> List[Tuple[str, str, int]]:
    """Identifies the most confused scam class pairs from the confusion matrix."""
    confusions = []
    n_classes = len(classes)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i][j] > 0:
                confusions.append((classes[i], classes[j], cm[i][j]))
    # Sort descending by count
    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:top_n]

def generate_baseline_report(
    results: Dict[str, Any],
    best_model_name: str,
    report_path: Path
):
    """Generates the docs/baseline_report.md markdown file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Scam Detection Models Evaluation Report\n\n")
        f.write("This report summarizes the performance, latency, and features of the trained baseline scam classification models.\n\n")
        
        # 1. Model Comparison Table
        f.write("## 1. Overall Performance Comparison\n\n")
        f.write("| Model Name | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | Weighted F1 | Training Time | Prediction Latency (per sample) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for name, res in results.items():
            metrics = res["metrics"]
            train_time = res["train_time"]
            pred_time = res["pred_time"]
            test_size = res["test_size"]
            latency_ms = (pred_time / test_size) * 1000.0 if test_size > 0 else 0
            
            f.write(
                f"| {name.capitalize()} | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
                f"{metrics['recall_macro']:.4f} | **{metrics['f1_macro']:.4f}** | {metrics['f1_weighted']:.4f} | "
                f"{train_time:.2f}s | {latency_ms:.3f} ms |\n"
            )
            
        f.write(f"\n**Best Model Selected**: `{best_model_name.capitalize()}` (based on highest Macro F1 score of **{results[best_model_name]['metrics']['f1_macro']:.4f}**).\n\n")
        
        # 2. Selected Model Hyperparameters
        f.write("## 2. Best Hyperparameters for Models\n\n")
        for name, res in results.items():
            best_params = res["best_params"]
            f.write(f"-   **{name.capitalize()}**:\n")
            for param, val in best_params.items():
                f.write(f"    - `{param}`: `{val}`\n")
        f.write("\n")

        # 3. Per-Class Metrics for Best Model
        f.write(f"## 3. Per-Class Metrics (Best Model: {best_model_name.capitalize()})\n\n")
        f.write("| Scam Category Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        best_metrics = results[best_model_name]["metrics"]
        for cls, metrics in sorted(best_metrics["class_metrics"].items()):
            f.write(
                f"| {cls} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['support']} |\n"
            )
        f.write("\n")
        
        # 4. Most Confused Scam Classes
        f.write(f"## 4. Top Misclassifications (Best Model: {best_model_name.capitalize()})\n\n")
        f.write("Below are the class pairs with the highest number of confusion instances (True label predicted as false label):\n\n")
        f.write("| True Scam Class | Predicted Class | Number of Confused Instances |\n")
        f.write("| :--- | :--- | :---: |\n")
        
        confusions = get_most_confused_classes(best_metrics["confusion_matrix"], best_metrics["classes"])
        for true_cls, pred_cls, count in confusions:
            f.write(f"| {true_cls} | {pred_cls} | {count} |\n")
        f.write("\n")
        
        # 5. Feature Importance & Key TF-IDF Predictors
        f.write("## 5. Top Predictive Features\n\n")
        f.write("This section shows the top terms (TF-IDF features) that are most strongly associated with each scam category.\n\n")
        
        for name, res in results.items():
            f.write(f"### Model: {name.capitalize()}\n\n")
            importance = res["feature_importance"]
            
            if "overall" in importance:
                # Random Forest overall importance
                f.write("#### Top 20 Overall Feature Importances:\n")
                f.write("| Rank | Feature Term | Importance Weight |\n")
                f.write("| :---: | :--- | :---: |\n")
                for idx, item in enumerate(importance["overall"][:20]):
                    f.write(f"| {idx+1} | {item['feature']} | {item['importance']:.6f} |\n")
                f.write("\n")
            else:
                # Per-class importances for Logistic, SVM, and NB
                # Show top 5 features for each class in columns/tables to keep it compact
                for cls, items in sorted(importance.items()):
                    if not items:
                        continue
                    f.write(f"#### Category: **{cls}**\n")
                    f.write("| Rank | Term / Ngram | Weight / Probability |\n")
                    f.write("| :---: | :--- | :---: |\n")
                    for idx, item in enumerate(items[:5]):
                        f.write(f"| {idx+1} | {item['feature']} | {item['importance']:.6f} |\n")
                    f.write("\n")
        f.write("\n")

def main():
    # Load configuration
    config = load_config()

    # Paths
    preprocessed_path = Path(project_root) / "data" / "final" / "preprocessed_dataset.pkl"
    models_dir = Path(project_root) / "models"
    metrics_path = models_dir / "baseline_metrics.json"
    report_path = Path(project_root) / "docs" / "baseline_report.md"

    # Ensure models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_split, test_split = load_preprocessed_dataset(preprocessed_path)
    
    X_train, y_train = train_split["texts"], train_split["labels"]
    X_test, y_test = test_split["texts"], test_split["labels"]

    logger.info(f"Loaded train samples: {len(X_train)} | test samples: {len(X_test)}")
    classes = sorted(list(set(y_train)))
    logger.info(f"Classification classes ({len(classes)}): {classes}")

    # 2. Define baseline models and hyperparameter grids
    # Parameter grids designed for performance and fast search
    models_to_train = {
        "logistic": {
            "clf": LogisticRegression(max_iter=1000, random_state=42),
            "grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "tfidf__max_df": [0.95],
                "clf__C": [0.1, 1.0, 10.0],
                "clf__class_weight": ["balanced"]
            }
        },
        "svm": {
            "clf": LinearSVC(dual=False, random_state=42),
            "grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "tfidf__max_df": [0.95],
                "clf__C": [0.1, 1.0, 10.0],
                "clf__class_weight": ["balanced"]
            }
        },
        "nb": {
            "clf": MultinomialNB(),
            "grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2, 5],
                "tfidf__max_df": [0.95],
                "clf__alpha": [0.1, 1.0]
            }
        },
        "rf": {
            "clf": RandomForestClassifier(random_state=42),
            "grid": {
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [2],
                "tfidf__max_df": [0.95],
                "clf__n_estimators": [100],
                "clf__class_weight": ["balanced"]
            }
        }
    }

    results = {}

    # 3. Train and evaluate each baseline model
    for name, setup in models_to_train.items():
        logger.info(f"=== Training model: {name.upper()} ===")
        best_pipeline, train_time = train_and_tune_model(
            model_name=name,
            clf=setup["clf"],
            param_grid=setup["grid"],
            X_train=X_train,
            y_train=y_train
        )
        
        # Evaluate
        metrics, pred_time = evaluate_model(best_pipeline, X_test, y_test, classes)
        
        # Extract best parameters
        best_params = {
            k: str(v) for k, v in best_pipeline.get_params().items()
            if any(k.startswith(prefix) for prefix in ["tfidf__", "clf__"])
        }
        
        # Extract features
        feature_importance = get_feature_importance(best_pipeline, classes)
        
        # Save results dictionary
        results[name] = {
            "pipeline": best_pipeline,
            "metrics": metrics,
            "best_params": best_params,
            "train_time": train_time,
            "pred_time": pred_time,
            "test_size": len(X_test),
            "feature_importance": feature_importance
        }
        
        # Save model pickle
        model_pkl_path = models_dir / f"baseline_{name}.pkl"
        logger.info(f"Saving model to {model_pkl_path}...")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(best_pipeline, f)

    # 4. Automatically select the best model using Macro F1
    best_model_name = max(results.keys(), key=lambda k: results[k]["metrics"]["f1_macro"])
    logger.info(f"Best model based on Macro F1: {best_model_name.upper()} (F1: {results[best_model_name]['metrics']['f1_macro']:.4f})")

    # 5. Save metrics dictionary (without the pipeline objects to keep it small and pure JSON)
    metrics_export = {
        "best_model": best_model_name,
        "models": {}
    }
    for name, res in results.items():
        metrics_export["models"][name] = {
            "metrics": res["metrics"],
            "best_params": res["best_params"],
            "train_time": res["train_time"],
            "pred_time": res["pred_time"]
        }
        
    logger.info(f"Saving metrics summary to {metrics_path}...")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=4)

    # 6. Generate baseline report
    logger.info(f"Generating report at {report_path}...")
    generate_baseline_report(results, best_model_name, report_path)
    logger.info("Baseline training and evaluation completed successfully.")

if __name__ == "__main__":
    main()
