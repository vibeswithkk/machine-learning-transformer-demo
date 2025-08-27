import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from functools import lru_cache
import threading

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
from sklearn.preprocessing import LabelEncoder

from .model import TransformerClassifier
from .utils import setup_logging, load_config, sanitize_input
from .exceptions import ModelNotFoundError, InferenceError, ValidationError


@dataclass
class PredictionResult:
    text: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    model_version: str
    timestamp: str


@dataclass
class BatchPredictionResult:
    results: List[PredictionResult]
    total_processing_time_ms: float
    throughput: float
    batch_size: int


class ModelCache:
    def __init__(self, max_size: int = 3):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.RLock()
    
    def get(self, model_path: str) -> Optional[torch.nn.Module]:
        with self.lock:
            if model_path in self.cache:
                self.cache.move_to_end(model_path)
                return self.cache[model_path]
            return None
    
    def put(self, model_path: str, model: torch.nn.Module) -> None:
        with self.lock:
            if model_path in self.cache:
                self.cache.move_to_end(model_path)
                self.cache[model_path] = model
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[model_path] = model


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        enable_caching: bool = True,
        max_batch_size: int = 32,
        max_sequence_length: int = 512,
        custom_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        log_level: int = logging.INFO,
        cache_size: int = 1000,
        use_asyncio: bool = False,
        max_concurrent_requests: int = 10
    ):
        self.logger = self._setup_logger(log_level)
        self.config = load_config(config_path) if config_path else {}
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.custom_tokenizer = custom_tokenizer
        self.cache_size = cache_size
        self.use_asyncio = use_asyncio
        
        self.model_cache = ModelCache() if enable_caching else None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.tokenizer = None
        self.model = None
        self.model_version = None
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prediction_cache = OrderedDict()
        self.cache_lock = threading.RLock()
        
        if use_asyncio:
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        else:
            self.semaphore = None
        
        self._load_model()
        self._validate_model()
    
    def _setup_logger(self, log_level: int) -> logging.Logger:
        logger = setup_logging(__name__)
        logger.setLevel(log_level)
        return logger
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 2 * 1024**3:
                return torch.device('cuda')
        
        return torch.device('cpu')
    
    def _load_model(self) -> None:
        try:
            if self.model_cache:
                cached_model = self.model_cache.get(str(self.model_path))
                if cached_model:
                    self.model = cached_model
                    if self.logger.level <= logging.INFO:
                        self.logger.info(f"Loaded model from cache: {self.model_path}")
                    return
            
            if not self.model_path.exists():
                raise ModelNotFoundError(f"Model file not found: {self.model_path}")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            model_config = checkpoint.get('config', {})
            self.model = TransformerClassifier(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_version = checkpoint.get('version', 'unknown')
            
            if 'classes' in checkpoint:
                self.class_names = checkpoint['classes']
                self.label_encoder.fit(self.class_names)
            elif 'label_encoder' in checkpoint:
                self.label_encoder = checkpoint['label_encoder']
                if hasattr(self.label_encoder, 'classes_'):
                    self.class_names = list(self.label_encoder.classes_)
            
            if self.custom_tokenizer:
                if isinstance(self.custom_tokenizer, str):
                    self.tokenizer = AutoTokenizer.from_pretrained(self.custom_tokenizer)
                else:
                    self.tokenizer = self.custom_tokenizer
            else:
                tokenizer_path = checkpoint.get('tokenizer_path', 'bert-base-uncased')
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            if self.model_cache:
                self.model_cache.put(str(self.model_path), self.model)
            
            if self.logger.level <= logging.INFO:
                self.logger.info(f"Successfully loaded model: {self.model_path}")
            
        except Exception as e:
            if self.logger.level <= logging.ERROR:
                self.logger.error(f"Failed to load model: {e}")
            raise ModelNotFoundError(f"Unable to load model: {e}")
    
    def _validate_model(self) -> None:
        if self.model is None:
            raise ValidationError("Model not loaded")
        
        if self.tokenizer is None:
            raise ValidationError("Tokenizer not loaded")
        
        try:
            sample_input = self.tokenizer(
                "test input",
                return_tensors="pt",
                max_length=self.max_sequence_length,
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                self.model(**sample_input)
            
            if self.logger.level <= logging.INFO:
                self.logger.info("Model validation successful")
            
        except Exception as e:
            if self.logger.level <= logging.ERROR:
                self.logger.error(f"Model validation failed: {e}")
            raise ValidationError(f"Model validation error: {e}")
    
    def _preprocess_text(self, text: str) -> torch.Tensor:
        sanitized_text = sanitize_input(text)
        
        if len(sanitized_text.strip()) == 0:
            raise ValidationError("Empty text input after sanitization")
        
        encoded = self.tokenizer(
            sanitized_text,
            return_tensors="pt",
            max_length=self.max_sequence_length,
            padding=True,
            truncation=True
        )
        
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def _get_cache_key(self, text: str) -> str:
        return f"{hash(text)}_{self.model_version}"
    
    def _check_cache(self, text: str) -> Optional[PredictionResult]:
        cache_key = self._get_cache_key(text)
        with self.cache_lock:
            if cache_key in self.prediction_cache:
                self.prediction_cache.move_to_end(cache_key)
                return self.prediction_cache[cache_key]
            return None
    
    def _update_cache(self, text: str, result: PredictionResult) -> None:
        cache_key = self._get_cache_key(text)
        with self.cache_lock:
            if cache_key in self.prediction_cache:
                self.prediction_cache.move_to_end(cache_key)
                self.prediction_cache[cache_key] = result
            else:
                if len(self.prediction_cache) >= self.cache_size:
                    self.prediction_cache.popitem(last=False)
                self.prediction_cache[cache_key] = result
    
    def predict_single(
        self,
        text: str,
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> PredictionResult:
        start_time = time.time()
        
        if use_cache:
            cached_result = self._check_cache(text)
            if cached_result:
                return cached_result
        
        try:
            inputs = self._preprocess_text(text)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                probabilities = F.softmax(logits, dim=-1)
                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                if self.class_names:
                    predicted_class = self.class_names[predicted_class_idx]
                    class_probs = {
                        self.class_names[i]: prob.item()
                        for i, prob in enumerate(probabilities[0])
                    }
                elif hasattr(self.label_encoder, 'classes_'):
                    predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                    class_probs = {
                        self.label_encoder.inverse_transform([i])[0]: prob.item()
                        for i, prob in enumerate(probabilities[0])
                    }
                else:
                    predicted_class = str(predicted_class_idx)
                    class_probs = {str(i): prob.item() for i, prob in enumerate(probabilities[0])}
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                text=text,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=class_probs if return_probabilities else {},
                processing_time_ms=processing_time_ms,
                model_version=self.model_version,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            if use_cache:
                self._update_cache(text, result)
            
            return result
            
        except Exception as e:
            if self.logger.level <= logging.ERROR:
                self.logger.error(f"Prediction failed for text: {text[:50]}... Error: {e}")
            raise InferenceError(f"Prediction failed: {e}")
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> BatchPredictionResult:
        start_time = time.time()
        
        if not texts:
            raise ValidationError("Empty text list provided")
        
        batch_size = batch_size or min(len(texts), self.max_batch_size)
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch_optimized(batch_texts, use_cache, return_probabilities)
            results.extend(batch_results)
        
        total_time_ms = (time.time() - start_time) * 1000
        throughput = len(results) / (total_time_ms / 1000) if total_time_ms > 0 else 0
        
        return BatchPredictionResult(
            results=results,
            total_processing_time_ms=total_time_ms,
            throughput=throughput,
            batch_size=len(texts)
        )
    
    def _process_batch_optimized(
        self,
        batch_texts: List[str],
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> List[PredictionResult]:
        batch_start_time = time.time()
        
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(batch_texts):
                cached_result = self._check_cache(text)
                if cached_result:
                    cached_results.append((i, cached_result))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = batch_texts
            uncached_indices = list(range(len(batch_texts)))
        
        results = [None] * len(batch_texts)
        
        for i, result in cached_results:
            results[i] = result
        
        if uncached_texts:
            try:
                sanitized_texts = [sanitize_input(text) for text in uncached_texts]
                
                encoded_batch = self.tokenizer(
                    sanitized_texts,
                    return_tensors="pt",
                    max_length=self.max_sequence_length,
                    padding=True,
                    truncation=True
                )
                
                encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}
                
                with torch.no_grad():
                    outputs = self.model(**encoded_batch)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    probabilities = F.softmax(logits, dim=-1)
                    predicted_classes = torch.argmax(probabilities, dim=-1)
                    confidences = torch.max(probabilities, dim=-1)[0]
                
                processing_time_per_item = (time.time() - batch_start_time) * 1000 / len(uncached_texts)
                
                for idx, (text_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                    predicted_class_idx = predicted_classes[idx].item()
                    confidence = confidences[idx].item()
                    
                    if self.class_names:
                        predicted_class = self.class_names[predicted_class_idx]
                        class_probs = {
                            self.class_names[i]: probabilities[idx][i].item()
                            for i in range(len(self.class_names))
                        } if return_probabilities else {}
                    elif hasattr(self.label_encoder, 'classes_'):
                        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                        class_probs = {
                            self.label_encoder.inverse_transform([i])[0]: probabilities[idx][i].item()
                            for i in range(len(self.label_encoder.classes_))
                        } if return_probabilities else {}
                    else:
                        predicted_class = str(predicted_class_idx)
                        class_probs = {
                            str(i): probabilities[idx][i].item()
                            for i in range(probabilities.shape[1])
                        } if return_probabilities else {}
                    
                    result = PredictionResult(
                        text=text,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        probabilities=class_probs,
                        processing_time_ms=processing_time_per_item,
                        model_version=self.model_version,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    results[text_idx] = result
                    
                    if use_cache:
                        self._update_cache(text, result)
                        
            except Exception as e:
                if self.logger.level <= logging.ERROR:
                    self.logger.error(f"Batch processing failed: {e}")
                
                for i, text in zip(uncached_indices, uncached_texts):
                    try:
                        result = self.predict_single(text, use_cache=False, return_probabilities=return_probabilities)
                        results[i] = result
                    except Exception as single_e:
                        if self.logger.level <= logging.WARNING:
                            self.logger.warning(f"Failed to process single text: {text[:50]}... Error: {single_e}")
        
        return [r for r in results if r is not None]
    
    async def predict_single_async(
        self,
        text: str,
        use_cache: bool = True,
        return_probabilities: bool = True
    ) -> PredictionResult:
        if self.use_asyncio and self.semaphore:
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    self.predict_single,
                    text,
                    use_cache,
                    return_probabilities
                )
        else:
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
    ) -> BatchPredictionResult:
        if self.use_asyncio and self.semaphore:
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
        else:
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
        return {
            "model_path": str(self.model_path),
            "model_version": self.model_version,
            "device": str(self.device),
            "max_sequence_length": self.max_sequence_length,
            "max_batch_size": self.max_batch_size,
            "num_classes": len(self.class_names) if self.class_names else (len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else "unknown"),
            "classes": self.class_names if self.class_names else (list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else []),
            "cache_size": len(self.prediction_cache),
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    def health_check(self) -> Dict[str, Any]:
        try:
            test_result = self.predict_single("health check test", use_cache=False)
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None,
                "device": str(self.device),
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cache_size": len(self.prediction_cache),
                "test_prediction_time_ms": test_result.processing_time_ms,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def clear_cache(self) -> None:
        with self.cache_lock:
            self.prediction_cache.clear()
        if self.logger.level <= logging.INFO:
            self.logger.info("Prediction cache cleared")
    
    def reload_model(self, model_path: Optional[str] = None) -> None:
        if model_path:
            self.model_path = Path(model_path)
        
        self.clear_cache()
        self._load_model()
        self._validate_model()
        if self.logger.level <= logging.INFO:
            self.logger.info(f"Model reloaded from: {self.model_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


def create_inference_engine(
    model_path: str,
    config_path: Optional[str] = None,
    **kwargs
) -> InferenceEngine:
    return InferenceEngine(
        model_path=model_path,
        config_path=config_path,
        **kwargs
    )


@lru_cache(maxsize=128)
def _cached_tokenize(tokenizer_name: str, text: str, max_length: int) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding=True,
        truncation=True
    )


def predict_sentiment(
    model_or_path: Union[InferenceEngine, str],
    text: str,
    **kwargs
) -> str:
    if isinstance(model_or_path, str):
        with create_inference_engine(model_or_path) as engine:
            result = engine.predict_single(text, **kwargs)
            return result.predicted_class
    else:
        result = model_or_path.predict_single(text, **kwargs)
        return result.predicted_class


def predict_sentiment_batch(
    model_or_path: Union[InferenceEngine, str],
    texts: List[str],
    **kwargs
) -> List[str]:
    if isinstance(model_or_path, str):
        with create_inference_engine(model_or_path) as engine:
            result = engine.predict_batch(texts, **kwargs)
            return [r.predicted_class for r in result.results]
    else:
        result = model_or_path.predict_batch(texts, **kwargs)
        return [r.predicted_class for r in result.results]