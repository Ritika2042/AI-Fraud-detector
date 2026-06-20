from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    TrainResponse, HealthResponse
)
from src.inference import ScamPredictor
from src.modeling.train import train_model
from src.logger import get_logger

logger = get_logger()
router = APIRouter()

def background_training_task() -> None:
    """Runs the training pipeline and updates the cached model in-memory."""
    try:
        logger.info("Asynchronous model retraining started...")
        train_model()
        logger.info("Asynchronous model retraining completed. Reloading model cache...")
        
        # Reload the cached model in-memory
        predictor = ScamPredictor()
        predictor.load_model()
        
        logger.info("Model cache successfully updated.")
    except Exception as e:
        logger.error(f"Error occurred during background model retraining: {e}")

def format_prediction_response(result: dict) -> PredictionResponse:
    """Maps internal inference results to the updated PredictionResponse schema."""
    pred = result["prediction"]
    conf = float(result["confidence"])
    
    # Compile reasons list for backwards compatibility
    reasons = []
    for category, details in result["evidence"].items():
        if details["detected"]:
            category_title = category.replace("_", " ").title()
            reasons.append(f"{category_title}: '{details['snippet']}'")
            
    if not reasons:
        reasons.append("No scam indicators detected. Conversation appears safe.")
        
    return PredictionResponse(
        prediction=pred,
        confidence=round(conf, 4),
        confidence_level=result["confidence_level"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        evidence=result["evidence"],
        top_keywords=result["top_keywords"],
        recommendation=result["recommendation"],
        
        # Backwards compatibility
        risk=result["risk_level"],
        reasons=reasons
    )

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a single conversation transcript",
    description="Analyzes the conversation transcript, predicts category, risk level, and triggers reasons."
)
async def predict(request: PredictionRequest):
    predictor = ScamPredictor()
    result = predictor.predict_single(request.conversation)
    
    if "error" in result:
        # If the model is missing, raise Service Unavailable (503)
        if "not loaded" in result["error"].lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result["error"]
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
        
    return format_prediction_response(result)

@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a list of conversation transcripts",
    description="Takes a batch of conversations and returns predictions, risk levels, and reasons for all of them."
)
async def predict_batch(request: BatchPredictionRequest):
    predictor = ScamPredictor()
    results = predictor.predict_batch(request.conversations)
    
    formatted_results = []
    for r in results:
        if "error" in r:
            if "not loaded" in r["error"].lower():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=r["error"]
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=r["error"]
            )
        formatted_results.append(format_prediction_response(r))
            
    return {"results": formatted_results}

@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger model retraining",
    description="Triggers the training pipeline in the background using the configured training dataset, and reloads the model once done."
)
async def train(background_tasks: BackgroundTasks):
    predictor = ScamPredictor()
    background_tasks.add_task(background_training_task)
    return TrainResponse(
        status="accepted",
        message="Model retraining has been scheduled in the background. The server remains online."
    )

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API status and model load state"
)
async def health():
    predictor = ScamPredictor()
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_model_loaded(),
        project=predictor.config.project_name
    )
