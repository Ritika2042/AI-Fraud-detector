import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from src.config import load_config
from src.logger import setup_logger
from src.data_pipeline.preprocessor import TextPreprocessor
from src.data_pipeline.validator import DataValidator, ALLOWED_LABELS
from src.modeling.transformer_model import TransformerTextDataset

def compute_metrics(eval_pred):
    """
    Computes classification performance metrics for evaluation steps.
    Calculates macro and weighted F1, precision, and recall alongside accuracy
    and logs the confusion matrix.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    # Calculate macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Calculate weighted-averaged metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    acc = accuracy_score(labels, predictions)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(f"\n--- Evaluation Confusion Matrix ---\n{cm}\n")
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted
    }

def train_transformer(config_path: str = None, model_name: str = "distilbert-base-uncased") -> None:
    """
    Executes the transformer fine-tuning pipeline:
    1. Loads configuration, sets logging, and checks datasets.
    2. Maps categories to numeric label IDs.
    3. Runs a stratified train-validation split.
    4. Tokenizes text sequences.
    5. Sets up HuggingFace Trainer with early stopping, mixed precision, and gradient accumulation.
    6. Trains, evaluates, and saves the final model.
    """
    # 1. Load config
    config = load_config(config_path)
    logger = setup_logger(config.logging)
    logger.info("--- Starting Transformer Fine-Tuning Pipeline ---")
    
    # 2. Read raw dataset
    raw_path = config.data.raw_data_path
    if not os.path.exists(raw_path):
        logger.error(f"Raw dataset not found at '{raw_path}'. Run scripts/generate_synthetic_data.py first.")
        raise FileNotFoundError(f"Raw dataset not found at '{raw_path}'")
        
    df = pd.read_csv(raw_path)
    
    # 3. Quality Validation
    validator = DataValidator(allowed_labels=ALLOWED_LABELS)
    if not validator.validate_training_df(df):
        logger.error("Dataset validation failed. Aborting transformer training.")
        raise ValueError("Dataset validation failed.")
        
    df = df.dropna(subset=["text", "label"])
    
    # 4. Preprocess Text
    logger.info("Preprocessing text data...")
    # Keep casing for transformers unless specifically using a lowercase-only model
    preprocessor = TextPreprocessor(lowercase=False)
    df = preprocessor.preprocess_df(df, text_column="text")
    
    # 5. Label Encoding
    classes = sorted(list(ALLOWED_LABELS))
    label_to_id = {label: i for i, label in enumerate(classes)}
    id_to_label = {i: label for i, label in enumerate(classes)}
    
    texts = df["text"].tolist()
    labels = [label_to_id[l] for l in df["label"]]
    
    # 6. Train-Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=labels
    )
    logger.info(f"Dataset Split: {len(X_train)} training records, {len(X_val)} validation records.")
    
    # 7. Tokenization
    logger.info(f"Initializing tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("Tokenizing text splits...")
    train_encodings = tokenizer(
        X_train,
        truncation=True,
        padding=True,
        max_length=256
    )
    val_encodings = tokenizer(
        X_val,
        truncation=True,
        padding=True,
        max_length=256
    )
    
    # Create PyTorch datasets
    train_dataset = TransformerTextDataset(train_encodings, y_train)
    val_dataset = TransformerTextDataset(val_encodings, y_val)
    
    # 8. Load Pretrained Sequence Classification Model
    logger.info(f"Loading pretrained model: '{model_name}'...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        id2label=id_to_label,
        label2id=label_to_id
    )
    
    # Enforce device allocation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model successfully loaded and allocated on device: {device}")
    
    # 9. Setup Training Arguments
    # Enable mixed precision (fp16) if GPU is available to speed up training
    fp16_enabled = torch.cuda.is_available()
    
    output_dir = "./models/transformer_checkpoints"
    logging_dir = "./logs/tensorboard"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=fp16_enabled,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=logging_dir,
        disable_tqdm=False
    )
    
    # 10. Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 11. Run Fine-Tuning
    logger.info("Executing training loop...")
    trainer.train()
    
    # 12. Evaluate Best Model
    logger.info("Running validation evaluation on the best checkpoint...")
    eval_metrics = trainer.evaluate()
    logger.info(f"Final Best Checkpoint Metrics: {eval_metrics}")
    
    # 13. Save Final Model & Tokenizer
    final_output_path = "./models/transformer_final"
    logger.info(f"Saving final fine-tuned model and tokenizer to '{final_output_path}'")
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    
    # Save classes text file for inference mappings
    with open(os.path.join(final_output_path, "classes.txt"), "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")
            
    logger.info("--- Transformer Fine-Tuning Completed Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune HuggingFace Transformer Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Base model name or path")
    args = parser.parse_args()
    
    train_transformer(args.config, args.model)
