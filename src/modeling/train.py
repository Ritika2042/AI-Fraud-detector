import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.config import load_config
from src.logger import setup_logger
from src.data_pipeline.preprocessor import TextPreprocessor
from src.data_pipeline.validator import DataValidator, ALLOWED_LABELS
from src.modeling.model import TfidfScamClassifier
from src.modeling.evaluate import evaluate_predictions, save_metrics

def train_model(config_path: str = None) -> None:
    """
    Executes the training pipeline:
    1. Loads configuration & logger.
    2. Reads and validates the raw dataset.
    3. Cleans and preprocesses conversation transcripts.
    4. Evaluates model stability using Stratified 5-Fold Cross Validation.
    5. Performs a stratified holdout train-test split.
    6. Trains the final TF-IDF + Classifier.
    7. Evaluates the classifier and saves model + metrics files.
    """
    # 1. Load configuration
    config = load_config(config_path)
    
    # 2. Setup logger
    logger = setup_logger(config.logging)
    logger.info("--- Starting Model Training Pipeline ---")
    
    # 3. Read dataset
    raw_path = config.data.raw_data_path
    if not os.path.exists(raw_path):
        logger.error(f"Raw dataset not found at '{raw_path}'. Please run the synthetic data generator first.")
        raise FileNotFoundError(f"Raw dataset not found at '{raw_path}'")
        
    logger.info(f"Loading raw dataset from {raw_path}")
    df = pd.read_csv(raw_path)
    
    # 4. Validate dataset schema and distributions
    validator = DataValidator(allowed_labels=ALLOWED_LABELS)
    if not validator.validate_training_df(df):
        logger.error("Dataset quality validation failed. Aborting training.")
        raise ValueError("Dataset quality validation failed.")
        
    # Drop rows containing missing text or labels to ensure valid training vectors
    df = df.dropna(subset=["text", "label"])
    
    # 5. Preprocess text
    logger.info("Cleaning and normalizing text transcripts...")
    preprocessor = TextPreprocessor(lowercase=config.model.tfidf.lowercase)
    df = preprocessor.preprocess_df(df, text_column="text")
    
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    
    # 6. Stratified Cross-Validation (K=5)
    logger.info("Evaluating model stability using Stratified 5-Fold Cross-Validation...")
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.data.random_state
    )
    
    # Convert lists to arrays for logical indexing
    X_arr = np.array(texts)
    y_arr = np.array(labels)
    
    fold_accuracies = []
    
    model_params = {
        "max_features": config.model.tfidf.max_features,
        "ngram_range": tuple(config.model.tfidf.ngram_range),
        "lowercase": config.model.tfidf.lowercase,
        "C": config.model.classifier.C,
        "max_iter": config.model.classifier.max_iter,
        "class_weight": config.model.classifier.class_weight,
        "random_state": config.model.classifier.random_state
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_train_f, X_val_f = X_arr[train_idx].tolist(), X_arr[val_idx].tolist()
        y_train_f, y_val_f = y_arr[train_idx].tolist(), y_arr[val_idx].tolist()
        
        fold_model = TfidfScamClassifier(**model_params)
        fold_model.fit(X_train_f, y_train_f)
        
        # Calculate accuracy on the validation fold
        val_preds = fold_model.predict(X_val_f)
        val_acc = np.mean(np.array(val_preds) == np.array(y_val_f))
        fold_accuracies.append(val_acc)
        logger.info(f"Fold {fold + 1}/5 - Validation Accuracy: {val_acc:.4f}")
        
    logger.info(f"Cross Validation Complete. Mean Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    
    # 7. Final Holdout Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=labels
    )
    logger.info(f"Final dataset split: {len(X_train)} training records, {len(X_test)} holdout test records.")
    
    # 8. Train Final Model
    logger.info("Training final model on full training set...")
    final_model = TfidfScamClassifier(**model_params)
    final_model.fit(X_train, y_train)
    
    # 9. Evaluate Final Model
    logger.info("Evaluating final model on holdout test set...")
    test_preds = final_model.predict(X_test)
    metrics = evaluate_predictions(y_test, test_preds, list(ALLOWED_LABELS))
    
    logger.info(f"Holdout Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Holdout Test Macro F1-Score: {metrics['macro_avg']['f1-score']:.4f}")
    
    # 10. Save Artifacts
    model_path = config.model.model_path
    metrics_path = config.model.metrics_path
    
    logger.info(f"Saving final model to '{model_path}'")
    final_model.save(model_path)
    
    logger.info(f"Saving holdout evaluation metrics to '{metrics_path}'")
    save_metrics(metrics, metrics_path)
    
    logger.info("--- Training Pipeline Completed Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Scam Classifier Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    
    train_model(args.config)
