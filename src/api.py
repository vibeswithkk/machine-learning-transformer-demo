#!/usr/bin/env python3
"""
REST API service for the machine learning transformer demo
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import logging

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from model import TransformerClassifier, ModelConfig
from inference import InferenceEngine, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Machine Learning Transformer Demo API",
    description="A production-ready API for text classification using transformer models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and inference engine
model = None
inference_engine = None

class TextInput(BaseModel):
    text: str = Field(..., example="This is a sample text for classification")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., example=["Text 1", "Text 2", "Text 3"])

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time_ms: float
    throughput: float

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    timestamp: str

def get_inference_engine():
    """Dependency to get the inference engine"""
    global inference_engine
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return inference_engine

def initialize_model():
    """Initialize the model and inference engine"""
    global model, inference_engine
    
    try:
        # Check if demo model exists
        model_path = "demo_model.pt"
        if not os.path.exists(model_path):
            logger.info("Creating demo model...")
            create_demo_model(model_path)
        
        # Initialize inference engine
        inference_engine = InferenceEngine(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_batch_size=32,
            max_sequence_length=128,
            cache_size=1000,
            log_level=logging.INFO
        )
        
        logger.info("Model and inference engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

def create_demo_model(model_path: str):
    """Create a demo model for testing purposes"""
    config = ModelConfig(
        vocab_size=1000,
        d_model=64,
        nhead=2,
        num_encoder_layers=2,
        dim_feedforward=128,
        max_position_embeddings=128,
        num_classes=3
    )
    
    model = TransformerClassifier(config)
    
    # Build vocabulary with sample texts
    sample_texts = [
        "This is a positive example with good sentiment",
        "Negative sentiment expressed in this text",
        "Neutral statement without strong emotion",
        "Another positive sentiment example",
        "Clearly negative feedback provided here",
        "Somewhat neutral opinion about this matter"
    ]
    model.build_tokenizer_vocab(sample_texts)
    
    # Save the model
    model.save_pretrained(model_path)
    logger.info(f"Demo model saved to {model_path}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model()

@app.get("/", summary="API Root", response_description="Welcome message")
async def root():
    """Root endpoint"""
    return {
        "message": "Machine Learning Transformer Demo API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse, summary="Health Check")
async def health_check():
    """Health check endpoint"""
    global inference_engine
    
    if inference_engine is None:
        health_status = {
            "status": "unhealthy",
            "model_loaded": False,
            "device": "unknown",
            "gpu_available": torch.cuda.is_available(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        try:
            health_info = inference_engine.health_check()
            health_status = {
                "status": health_info.get("status", "unknown"),
                "model_loaded": health_info.get("model_loaded", False),
                "device": health_info.get("device", "unknown"),
                "gpu_available": health_info.get("gpu_available", torch.cuda.is_available()),
                "timestamp": health_info.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            }
        except Exception as e:
            health_status = {
                "status": "unhealthy",
                "model_loaded": True,
                "device": "unknown",
                "gpu_available": torch.cuda.is_available(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
    
    return health_status

@app.post("/predict", response_model=PredictionResponse, summary="Single Text Prediction")
async def predict(text_input: TextInput, engine: InferenceEngine = Depends(get_inference_engine)):
    """Predict sentiment for a single text"""
    try:
        start_time = time.time()
        
        result = engine.predict_single(text_input.text)
        
        response = PredictionResponse(
            prediction=result.predicted_class,
            confidence=result.confidence,
            probabilities=result.probabilities,
            processing_time_ms=result.processing_time_ms
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch Text Prediction")
async def predict_batch(batch_input: BatchTextInput, engine: InferenceEngine = Depends(get_inference_engine)):
    """Predict sentiment for a batch of texts"""
    try:
        start_time = time.time()
        
        batch_result = engine.predict_batch(batch_input.texts)
        
        predictions = [
            PredictionResponse(
                prediction=pred.predicted_class,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                processing_time_ms=pred.processing_time_ms
            )
            for pred in batch_result.results
        ]
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=batch_result.total_processing_time_ms,
            throughput=batch_result.throughput
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info", summary="Model Information")
async def model_info(engine: InferenceEngine = Depends(get_inference_engine)):
    """Get model information"""
    try:
        info = engine.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)