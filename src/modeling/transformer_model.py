import os
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.modeling.base import BaseScamClassifier
from src.logger import get_logger

logger = get_logger()

class TransformerTextDataset(Dataset):
    """PyTorch Dataset wrapper for Hugging Face tokenized encodings."""
    
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values())))

class TransformerScamClassifier(BaseScamClassifier):
    """
    Scam Classifier implementing a HuggingFace Transformer model.
    Fully compliant with the BaseScamClassifier clean architecture interface.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        num_labels: int = 10,
        max_length: int = 256,
        classes: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.classes_ = classes if classes is not None else []
        
        # Detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Transformer model on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels
        ).to(self.device)

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Implementation interface only. 
        For full fine-tuning with training loops, use train_transformer.py.
        """
        raise NotImplementedError(
            "To train/fine-tune the transformer model, run the training script "
            "'src/modeling/train_transformer.py' which handles training args, checkpoints, and evaluation."
        )

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts scam categories for a list of texts."""
        if not self.classes_:
            raise ValueError("Class labels must be defined before running prediction.")
            
        self.model.eval()
        predictions = []
        
        # Process in batch to prevent GPU memory overflow
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend([self.classes_[p] for p in preds])
                
        return predictions

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predicts class probabilities for a list of texts."""
        if not self.classes_:
            raise ValueError("Class labels must be defined before running prediction.")

        self.model.eval()
        probabilities = []
        
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                # Apply Softmax to get probability distribution
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                
                for row in probs:
                    prob_dict = {self.classes_[idx]: float(row[idx]) for idx in range(len(self.classes_))}
                    probabilities.append(prob_dict)
                    
        return probabilities

    def explain(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Extracts token importances using a gradient-based approach (Integrated Gradients baseline).
        """
        self.model.eval()
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get embeddings layer
        embeddings = self.model.get_input_embeddings()
        
        # We want to trace gradients with respect to input embeddings
        input_embeds = embeddings(input_ids).detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        pred_class = torch.argmax(logits, dim=-1).item()
        
        # Backward pass with respect to predicted class logit
        score = logits[0, pred_class]
        score.backward()
        
        # Saliency map is the L2 norm of the gradient at each token position
        saliency = torch.norm(input_embeds.grad[0], dim=-1).cpu().numpy()
        
        # Filter out special tokens (like [CLS], [SEP], [PAD]) and format
        explanations = []
        special_tokens = {self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token}
        
        for idx, (token, val) in enumerate(zip(tokens, saliency)):
            if token not in special_tokens and not token.startswith("##") and val > 0:
                explanations.append({
                    "word": token,
                    "weight": float(val)
                })
                
        # Sort and return top_n
        explanations = sorted(explanations, key=lambda x: x["weight"], reverse=True)
        return explanations[:top_n]

    def save(self, file_path: str) -> None:
        """Saves model and tokenizer to a directory."""
        # HuggingFace models are saved as a directory, not a single file
        os.makedirs(file_path, exist_ok=True)
        self.model.save_pretrained(file_path)
        self.tokenizer.save_pretrained(file_path)
        
        # Save class labels list to model folder
        labels_file = os.path.join(file_path, "classes.txt")
        with open(labels_file, "w", encoding="utf-8") as f:
            for cls in self.classes_:
                f.write(f"{cls}\n")
        logger.info(f"Transformer model and classes configuration saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "TransformerScamClassifier":
        """Loads and returns an instance from the saved directory."""
        # Load class labels
        labels_file = os.path.join(file_path, "classes.txt")
        classes = []
        if os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
                
        obj = cls(
            model_name_or_path=file_path,
            num_labels=len(classes) if classes else 10,
            classes=classes
        )
        return obj
