import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.modeling.base import BaseScamClassifier

class TfidfScamClassifier(BaseScamClassifier):
    """
    Concrete implementation of BaseScamClassifier using TF-IDF and Logistic Regression.
    Provides fast training, low-latency inference, and exact keyword explainability.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        lowercase: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Trains both the TF-IDF Vectorizer and the Logistic Regression classifier."""
        if not texts or not labels:
            raise ValueError("Training inputs (texts and labels) cannot be empty.")
            
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = list(self.classifier.classes_)

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts the scam category for a list of conversation transcripts."""
        if not self.classes_:
            raise ValueError("Model has not been trained yet. Call fit() before predicting.")
        
        X = self.vectorizer.transform(texts)
        return list(self.classifier.predict(X))

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predicts class probabilities for a list of conversation transcripts."""
        if not self.classes_:
            raise ValueError("Model has not been trained yet. Call fit() before predicting probabilities.")

        X = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(X)
        
        results = []
        for prob_row in probs:
            prob_dict = {self.classes_[i]: float(prob_row[i]) for i in range(len(self.classes_))}
            results.append(prob_dict)
        return results

    def explain(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Explains predictions by identifying which tokens in the input text had the highest
        positive weight contribution (TF-IDF value * model coefficient) toward the predicted class.
        """
        if not self.classes_:
            raise ValueError("Model has not been trained yet. Call fit() before running explanations.")

        # 1. Transform text to tf-idf vector
        v = self.vectorizer.transform([text])
        non_zero_indices = v.nonzero()[1]
        
        if len(non_zero_indices) == 0:
            return []

        # 2. Find the predicted label
        pred_label = self.predict([text])[0]
        class_idx = self.classes_.index(pred_label)

        # 3. Retrieve coefficient contributions
        # If binary class, LogisticRegression.coef_ has shape (1, n_features)
        # If multiclass, LogisticRegression.coef_ has shape (n_classes, n_features)
        is_multiclass = len(self.classes_) > 2
        feature_names = self.vectorizer.get_feature_names_out()

        explanations = []
        for idx in non_zero_indices:
            tfidf_val = v[0, idx]
            if is_multiclass:
                coef_val = self.classifier.coef_[class_idx, idx]
            else:
                raw_coef = self.classifier.coef_[0, idx]
                coef_val = raw_coef if class_idx == 1 else -raw_coef

            contribution = float(tfidf_val * coef_val)
            if contribution > 0:  # Only show tokens that actively support this classification
                explanations.append({
                    "word": str(feature_names[idx]),
                    "weight": contribution
                })

        # Sort descending by contribution weight
        explanations = sorted(explanations, key=lambda x: x["weight"], reverse=True)
        return explanations[:top_n]

    def save(self, file_path: str) -> None:
        """Saves the pipeline state (vectorizer, classifier, classes) using pickle."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "classes_": self.classes_,
            "params": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "lowercase": self.lowercase,
                "C": self.C,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
                "random_state": self.random_state
            }
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, file_path: str) -> "TfidfScamClassifier":
        """Loads and returns an instance of TfidfScamClassifier from pickle."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
            
        with open(path, "rb") as f:
            state = pickle.load(f)

        params = state["params"]
        obj = cls(**params)
        obj.vectorizer = state["vectorizer"]
        obj.classifier = state["classifier"]
        obj.classes_ = state["classes_"]
        return obj
