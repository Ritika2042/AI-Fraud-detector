import os
import threading
from typing import Dict, Any, List, Optional
from src.config import load_config
from src.logger import get_logger
from src.data_pipeline.preprocessor import TextPreprocessor
from src.modeling.model import TfidfScamClassifier

# Import new Explainable AI layer components
from src.reasoning.evidence_extractor import EvidenceExtractor
from src.reasoning.risk_engine import RiskEngine
from src.reasoning.recommendation_engine import RecommendationEngine
from src.reasoning.confidence_calibrator import ConfidenceCalibrator

logger = get_logger()

class ScamPredictor:
    """
    Thread-safe Singleton inference class.
    Loads and caches the model, runs predictions, and extracts explanations.
    """
    _instance: Optional["ScamPredictor"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ScamPredictor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
            
        self.config = load_config(config_path)
        self.preprocessor = TextPreprocessor(lowercase=self.config.model.tfidf.lowercase)
        self.model_path = self.config.model.model_path
        self.model: Optional[TfidfScamClassifier] = None
        self.model_lock = threading.Lock()
        
        # Instantiate Explainable AI components
        self.evidence_extractor = EvidenceExtractor()
        self.risk_engine = RiskEngine()
        self.recommendation_engine = RecommendationEngine()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        self.load_model()
        self._initialized = True

    def load_model(self) -> None:
        """
        Loads the pickled model state from disk.
        Uses a lock to prevent concurrent reading/writing of the model instance.
        """
        with self.model_lock:
            if not os.path.exists(self.model_path):
                logger.warning(
                    f"Model file not found at '{self.model_path}'. "
                    "Inference will return errors until the training pipeline runs successfully."
                )
                self.model = None
                return
                
            try:
                logger.info(f"Loading scam detection classifier from '{self.model_path}'")
                self.model = TfidfScamClassifier.load(self.model_path)
                logger.info("Classifier successfully loaded.")
            except Exception as e:
                logger.error(f"Failed to load model from '{self.model_path}': {e}")
                self.model = None

    def is_model_loaded(self) -> bool:
        """Checks if the classifier model is loaded and ready for predictions."""
        with self.model_lock:
            return self.model is not None

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Preprocesses and predicts the scam class for a single transcript.
        Runs it through the explainable AI layer to retrieve risk scores, levels,
        and safety recommendations.
        """
        if not text.strip():
            return {"error": "Input text is empty."}
            
        cleaned_text = self.preprocessor.clean_text(text)
        
        with self.model_lock:
            if self.model is None:
                return {"error": "Scam classifier model is not loaded. Run training first."}
                
            try:
                # Run ML model inference
                pred_label = self.model.predict([cleaned_text])[0]
                prob_dict = self.model.predict_proba([cleaned_text])[0]
                confidence = float(prob_dict[pred_label])
                explanations = self.model.explain(cleaned_text, top_n=5)
                
                # Apply Explainable AI layer
                evidence_res = self.evidence_extractor.extract_evidence(text)
                risk_res = self.risk_engine.compute_risk(evidence_res["evidence"])
                confidence_level = self.confidence_calibrator.calibrate(confidence)
                recommendation = self.recommendation_engine.get_recommendation(pred_label)
                
                return {
                    "prediction": pred_label,
                    "confidence": confidence,
                    "confidence_level": confidence_level,
                    "risk_score": risk_res["risk_score"],
                    "risk_level": risk_res["risk_level"],
                    "evidence": evidence_res["evidence"],
                    "top_keywords": explanations,
                    "recommendation": recommendation,
                    
                    # Backwards compatibility fields
                    "probabilities": prob_dict,
                    "reasoning_analysis": evidence_res,
                    "explanations": explanations
                }
            except Exception as e:
                logger.error(f"Inference prediction error: {e}")
                return {"error": f"Prediction failed: {e}"}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Preprocesses and predicts scam labels for a batch of transcripts.
        """
        if not texts:
            return []
            
        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        
        with self.model_lock:
            if self.model is None:
                return [{"error": "Scam classifier model is not loaded."} for _ in texts]
                
            try:
                preds = self.model.predict(cleaned_texts)
                probs = self.model.predict_proba(cleaned_texts)
                
                results = []
                for idx, (pred, prob_dict) in enumerate(zip(preds, probs)):
                    confidence = float(prob_dict[pred])
                    explanation = self.model.explain(cleaned_texts[idx], top_n=5)
                    
                    # Apply Explainable AI layer
                    evidence_res = self.evidence_extractor.extract_evidence(texts[idx])
                    risk_res = self.risk_engine.compute_risk(evidence_res["evidence"])
                    confidence_level = self.confidence_calibrator.calibrate(confidence)
                    recommendation = self.recommendation_engine.get_recommendation(pred)
                    
                    results.append({
                        "prediction": pred,
                        "confidence": confidence,
                        "confidence_level": confidence_level,
                        "risk_score": risk_res["risk_score"],
                        "risk_level": risk_res["risk_level"],
                        "evidence": evidence_res["evidence"],
                        "top_keywords": explanation,
                        "recommendation": recommendation,
                        
                        # Backwards compatibility fields
                        "probabilities": prob_dict,
                        "reasoning_analysis": evidence_res,
                        "explanations": explanation
                    })
                return results
            except Exception as e:
                logger.error(f"Inference batch prediction error: {e}")
                return [{"error": f"Batch prediction failed: {e}"} for _ in texts]
