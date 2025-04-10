from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Interview Emotion Analyzer",
    description="API for analyzing emotions in interview videos using multiple modalities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting Interview Emotion Analyzer API")
    # Add any additional startup initialization here

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Interview Emotion Analyzer API")
    # Add any cleanup code here

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 