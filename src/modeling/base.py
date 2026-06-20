from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseScamClassifier(ABC):
    """Abstract Base Class representing a Scam Classification Model."""

    @abstractmethod
    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Trains the classifier on a list of texts and their corresponding labels.
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predicts the class label for each text in the provided list.
        """
        pass

    @abstractmethod
    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predicts class probabilities for each text in the provided list.
        Returns a list of dicts mapping class labels to their respective probabilities.
        """
        pass

    @abstractmethod
    def explain(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Explains the prediction by identifying the top keywords contributing
        to the predicted class.
        Returns a list of dicts with 'word' and 'weight' keys.
        """
        pass

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Serializes and saves the model artifacts to the specified file path.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> "BaseScamClassifier":
        """
        Loads the serialized model artifacts from file and returns an instance.
        """
        pass
