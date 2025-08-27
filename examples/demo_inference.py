#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass
from functools import lru_cache

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model import TransformerClassifier, ModelConfig
from src.inference import InferenceEngine, PredictionResult


@dataclass
class BatchInferenceResult:
    predictions: List[PredictionResult]
    total_processing_time_ms: float
    throughput: float
    batch_size: int


class ProductionInferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_batch_size: int = 32,
        max_sequence_length: int = 512,
        cache_size: int = 1000,
        max_concurrent_requests: int = 10,
        log_level: int = logging.INFO
    ):
        self.model_path = Path(model_path)
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.cache_size = cache_size
        self.max_concurrent_requests = max_concurrent_requests
        self.log_level = log_level
        
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.engine = InferenceEngine(
            model_path=str(self.model_path),
            device=str(self.device),
            max_batch_size=self.max_batch_size,
            max_sequence_length=self.max_sequence_length,
            cache_size=self.cache_size,
            use_asyncio=True,
            max_concurrent_requests=self.max_concurrent_requests,
            log_level=self.log_level
        )
        
        # Thread safety
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Performance tracking
        self.prediction_cache = OrderedDict()
        self.stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0,
            "average_response_time": 0.0
        }
    
    def predict_single(
        self,
        text: str,
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> PredictionResult:
        with self.lock:
            start_time = time.time()
            
            try:
                result = self.engine.predict_single(
                    text=text,
                    use_cache=use_cache,
                    return_probabilities=return_probabilities
                )
                
                # Update statistics
                processing_time = time.time() - start_time
                self.stats["total_predictions"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["average_response_time"] = (
                    self.stats["total_processing_time"] / self.stats["total_predictions"]
                )
                
                return result
                
            except Exception as e:
                logging.error(f"Inference failed for text: {text[:50]}... Error: {str(e)}")
                raise
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> BatchInferenceResult:
        start_time = time.time()
        
        try:
            batch_result = self.engine.predict_batch(
                texts=texts,
                batch_size=batch_size or self.max_batch_size,
                use_cache=use_cache,
                return_probabilities=return_probabilities
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_predictions"] += len(texts)
            self.stats["total_processing_time"] += processing_time
            self.stats["average_response_time"] = (
                self.stats["total_processing_time"] / self.stats["total_predictions"]
            )
            
            return BatchInferenceResult(
                predictions=batch_result.results,
                total_processing_time_ms=batch_result.total_processing_time_ms,
                throughput=batch_result.throughput,
                batch_size=batch_result.batch_size
            )
            
        except Exception as e:
            logging.error(f"Batch inference failed for {len(texts)} texts. Error: {str(e)}")
            raise
    
    async def predict_single_async(
        self,
        text: str,
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> PredictionResult:
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.predict_single,
                text,
                use_cache,
                return_probabilities
            )
    
    async def predict_batch_async(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> BatchInferenceResult:
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.predict_batch,
                texts,
                batch_size,
                use_cache,
                return_probabilities
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        return self.engine.get_model_info()
    
    def health_check(self) -> Dict[str, Any]:
        return self.engine.health_check()
    
    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            stats_copy = self.stats.copy()
            if self.stats["total_predictions"] > 0:
                stats_copy["cache_hit_rate"] = (
                    self.stats["cache_hits"] / self.stats["total_predictions"]
                )
            else:
                stats_copy["cache_hit_rate"] = 0.0
            return stats_copy
    
    def clear_cache(self) -> None:
        with self.lock:
            self.engine.clear_cache()
            self.prediction_cache.clear()
            self.stats["cache_hits"] = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


def create_demo_model():
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
    
    return model


def save_demo_model(model, model_path):
    """Save the demo model to disk"""
    model.save_pretrained(model_path)
    print(f"Demo model saved to {model_path}")


def main():
    """Main function demonstrating the production inference engine"""
    print("Production Inference Engine Demo")
    print("=" * 40)
    
    # Create and save demo model if it doesn't exist
    model_path = "demo_model.pt"
    if not os.path.exists(model_path):
        print("Creating demo model...")
        model = create_demo_model()
        save_demo_model(model, model_path)
    
    # Initialize production inference engine
    print("Initializing production inference engine...")
    engine = ProductionInferenceEngine(
        model_path=model_path,
        max_batch_size=16,
        max_sequence_length=64,
        cache_size=100,
        max_concurrent_requests=5
    )
    
    # Perform health check
    print("\nPerforming health check...")
    health = engine.health_check()
    print(f"Health status: {health['status']}")
    print(f"Device: {health['device']}")
    print(f"GPU available: {health['gpu_available']}")
    
    # Single prediction example
    print("\nSingle prediction example:")
    text = "This is a great product with excellent features!"
    result = engine.predict_single(text)
    print(f"Text: {result.text}")
    print(f"Predicted class: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    
    # Batch prediction example
    print("\nBatch prediction example:")
    texts = [
        "This is a positive review of the product",
        "I'm not satisfied with this service",
        "The product is okay, nothing special",
        "Amazing quality and fast delivery!",
        "Poor customer support experience"
    ]
    
    batch_result = engine.predict_batch(texts)
    print(f"Batch size: {batch_result.batch_size}")
    print(f"Total processing time: {batch_result.total_processing_time_ms:.2f}ms")
    print(f"Throughput: {batch_result.throughput:.2f} samples/sec")
    
    print("\nIndividual predictions:")
    for i, pred in enumerate(batch_result.predictions):
        print(f"  {i+1}. '{pred.text[:30]}...' -> {pred.predicted_class} ({pred.confidence:.4f})")
    
    # Performance statistics
    print("\nPerformance statistics:")
    stats = engine.get_statistics()
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Average response time: {stats['average_response_time']*1000:.2f}ms")
    
    # Model information
    print("\nModel information:")
    info = engine.get_model_info()
    print(f"Model version: {info['model_version']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Max sequence length: {info['max_sequence_length']}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()