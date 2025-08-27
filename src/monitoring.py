#!/usr/bin/env python3
"""
Model monitoring and observability module
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from datetime import datetime
import numpy as np

from prometheus_client import Counter, Histogram, Gauge, Summary


@dataclass
class PredictionMetrics:
    """Data class for prediction metrics"""
    timestamp: float
    processing_time_ms: float
    text_length: int
    model_version: str
    predicted_class: str
    confidence: float
    cache_hit: bool


class ModelMonitor:
    """Model monitoring and observability class"""
    
    def __init__(self, model_name: str = "transformer_classifier"):
        self.model_name = model_name
        self.lock = threading.Lock()
        
        # Prometheus metrics
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions',
            ['model_name', 'predicted_class']
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name']
        )
        
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Prediction confidence scores',
            ['model_name'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.cache_hit_ratio = Gauge(
            'model_cache_hit_ratio',
            'Cache hit ratio',
            ['model_name']
        )
        
        self.model_performance = Summary(
            'model_performance_summary',
            'Model performance summary',
            ['model_name']
        )
        
        # In-memory metrics storage
        self.metrics_history = deque(maxlen=10000)  # Keep last 10,000 predictions
        self.cache_hits = 0
        self.total_predictions = 0
        
        # Performance tracking
        self.performance_stats = {
            'avg_latency': 0.0,
            'avg_confidence': 0.0,
            'class_distribution': defaultdict(int),
            'text_length_stats': {'min': float('inf'), 'max': 0, 'sum': 0, 'count': 0}
        }
    
    def record_prediction(self, metrics: PredictionMetrics):
        """Record prediction metrics"""
        with self.lock:
            # Update Prometheus metrics
            self.prediction_counter.labels(
                model_name=self.model_name,
                predicted_class=metrics.predicted_class
            ).inc()
            
            self.prediction_latency.labels(
                model_name=self.model_name
            ).observe(metrics.processing_time_ms / 1000.0)  # Convert to seconds
            
            self.prediction_confidence.labels(
                model_name=self.model_name
            ).observe(metrics.confidence)
            
            # Update cache hit ratio
            if metrics.cache_hit:
                self.cache_hits += 1
            self.total_predictions += 1
            
            cache_hit_ratio = self.cache_hits / self.total_predictions if self.total_predictions > 0 else 0
            self.cache_hit_ratio.labels(model_name=self.model_name).set(cache_hit_ratio)
            
            # Store metrics in history
            self.metrics_history.append(metrics)
            
            # Update performance statistics
            self._update_performance_stats(metrics)
    
    def _update_performance_stats(self, metrics: PredictionMetrics):
        """Update internal performance statistics"""
        # Update latency stats
        if self.performance_stats['avg_latency'] == 0:
            self.performance_stats['avg_latency'] = metrics.processing_time_ms
        else:
            self.performance_stats['avg_latency'] = (
                self.performance_stats['avg_latency'] * 0.9 + 
                metrics.processing_time_ms * 0.1
            )
            
        # Update confidence stats
        if self.performance_stats['avg_confidence'] == 0:
            self.performance_stats['avg_confidence'] = metrics.confidence
        else:
            self.performance_stats['avg_confidence'] = (
                self.performance_stats['avg_confidence'] * 0.9 + 
                metrics.confidence * 0.1
            )
            
        # Update class distribution
        self.performance_stats['class_distribution'][metrics.predicted_class] += 1
        
        # Update text length stats
        text_stats = self.performance_stats['text_length_stats']
        text_stats['min'] = min(text_stats['min'], metrics.text_length)
        text_stats['max'] = max(text_stats['max'], metrics.text_length)
        text_stats['sum'] += metrics.text_length
        text_stats['count'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        with self.lock:
            text_stats = self.performance_stats['text_length_stats']
            avg_text_length = (
                text_stats['sum'] / text_stats['count'] 
                if text_stats['count'] > 0 else 0
            )
            
            # Calculate class distribution percentages
            total_class_preds = sum(self.performance_stats['class_distribution'].values())
            class_distribution_pct = {
                cls: count / total_class_preds 
                for cls, count in self.performance_stats['class_distribution'].items()
            } if total_class_preds > 0 else {}
            
            return {
                'model_name': self.model_name,
                'total_predictions': self.total_predictions,
                'cache_hit_ratio': self.cache_hits / self.total_predictions if self.total_predictions > 0 else 0,
                'average_latency_ms': self.performance_stats['avg_latency'],
                'average_confidence': self.performance_stats['avg_confidence'],
                'class_distribution': class_distribution_pct,
                'text_length_stats': {
                    'min': text_stats['min'] if text_stats['min'] != float('inf') else 0,
                    'max': text_stats['max'],
                    'average': avg_text_length
                },
                'metrics_history_size': len(self.metrics_history)
            }
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent prediction metrics"""
        with self.lock:
            recent = list(self.metrics_history)[-limit:]
            return [asdict(metric) for metric in recent]
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring"""
        return {
            'performance_summary': self.get_performance_summary(),
            'recent_predictions': self.get_recent_predictions(50),
            'timestamp': datetime.utcnow().isoformat()
        }


class PredictionLogger:
    """Structured prediction logging class"""
    
    def __init__(self, log_file: str = "predictions.log"):
        self.logger = logging.getLogger("prediction_logger")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def log_prediction(self, input_text: str, prediction: str, 
                      confidence: float, processing_time_ms: float,
                      model_version: str, cache_hit: bool = False):
        """Log a prediction event"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'input_text': input_text[:100] + '...' if len(input_text) > 100 else input_text,
            'prediction': prediction,
            'confidence': confidence,
            'processing_time_ms': processing_time_ms,
            'model_version': model_version,
            'cache_hit': cache_hit
        }
        
        self.logger.info(json.dumps(log_entry))


# Global monitor instance
model_monitor = ModelMonitor()
prediction_logger = PredictionLogger()


def get_model_monitor() -> ModelMonitor:
    """Get the global model monitor instance"""
    return model_monitor


def get_prediction_logger() -> PredictionLogger:
    """Get the global prediction logger instance"""
    return prediction_logger


# Example usage
if __name__ == "__main__":
    # Create monitor
    monitor = ModelMonitor("test_model")
    
    # Simulate some predictions
    for i in range(100):
        metrics = PredictionMetrics(
            timestamp=time.time(),
            processing_time_ms=np.random.uniform(10, 100),
            text_length=np.random.randint(10, 200),
            model_version="1.0.0",
            predicted_class=np.random.choice(["positive", "negative", "neutral"]),
            confidence=np.random.uniform(0.5, 1.0),
            cache_hit=np.random.choice([True, False], p=[0.3, 0.7])
        )
        
        monitor.record_prediction(metrics)
        
        if i % 20 == 0:
            summary = monitor.get_performance_summary()
            print(f"Performance summary at step {i}:")
            print(f"  Total predictions: {summary['total_predictions']}")
            print(f"  Cache hit ratio: {summary['cache_hit_ratio']:.2f}")
            print(f"  Average latency: {summary['average_latency_ms']:.2f}ms")
            print(f"  Average confidence: {summary['average_confidence']:.2f}")