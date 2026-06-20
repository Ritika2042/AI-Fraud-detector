import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from src.config import load_config
from src.logger import setup_logger
from src.api.routes import router as api_router
from src.inference import ScamPredictor

# 1. Load system settings
config = load_config()

# 2. Configure application logger
logger = setup_logger(config.logging)
logger.info(f"Initializing {config.project_name} API Server...")

# 3. Create FastAPI app
app = FastAPI(
    title="AI Scam Detection System API",
    description=(
        "Production-grade REST API for classifying conversation transcripts into safe "
        "or scam categories. Powered by a TF-IDF & Logistic Regression classification model."
    ),
    version="1.0.0"
)

# 4. Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Register API Routing Submodule
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("API server is starting up...")
    # Pre-initialize ScamPredictor to trigger model load on startup
    predictor = ScamPredictor()
    if predictor.is_model_loaded():
        logger.info("Model loaded successfully during server startup.")
    else:
        logger.warning("API started without a pre-loaded model. Retraining via /api/v1/train is required.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API server is shutting down...")

@app.get("/", include_in_schema=False)
async def root():
    """Redirects base path to Interactive OpenAPI Swagger Documentation."""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    # Allow running directly via 'python src/api/main.py'
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    )
