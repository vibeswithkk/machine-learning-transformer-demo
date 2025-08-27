#!/usr/bin/env python3
"""
Model registry for versioning and management
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil
from pathlib import Path
import torch


@dataclass
class ModelMetadata:
    """Data class for model metadata"""
    model_id: str
    name: str
    version: str
    created_at: str
    framework: str
    architecture: str
    input_shape: Dict[str, Any]
    output_shape: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    description: str
    file_path: str
    file_size: int
    checksum: str
    author: str = "anonymous"
    dependencies: List[str] = None


class ModelRegistry:
    """Model registry for versioning and management"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_file = self.registry_path / "models.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing models
        self.models = self._load_models()
    
    def _load_models(self) -> Dict[str, ModelMetadata]:
        """Load models from registry file"""
        if not self.models_file.exists():
            return {}
            
        try:
            with open(self.models_file, 'r') as f:
                data = json.load(f)
                models = {}
                for model_id, model_data in data.items():
                    models[model_id] = ModelMetadata(**model_data)
                return models
        except Exception as e:
            print(f"Warning: Could not load models from registry: {e}")
            return {}
    
    def _save_models(self):
        """Save models to registry file"""
        try:
            data = {model_id: asdict(model) for model_id, model in self.models.items()}
            with open(self.models_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving models to registry: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def register_model(self, 
                      model_path: str,
                      name: str,
                      version: str,
                      framework: str = "pytorch",
                      architecture: str = "transformer",
                      input_shape: Dict[str, Any] = None,
                      output_shape: Dict[str, Any] = None,
                      metrics: Dict[str, float] = None,
                      tags: List[str] = None,
                      description: str = "",
                      author: str = "anonymous",
                      dependencies: List[str] = None) -> str:
        """Register a model in the registry"""
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate unique model ID
        model_id = f"{name}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Copy model to registry
        model_filename = f"{model_id}.pt"
        registry_model_path = self.models_dir / model_filename
        shutil.copy2(model_path, registry_model_path)
        
        # Calculate file size and checksum
        file_size = os.path.getsize(registry_model_path)
        checksum = self._calculate_checksum(str(registry_model_path))
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            framework=framework,
            architecture=architecture,
            input_shape=input_shape or {},
            output_shape=output_shape or {},
            metrics=metrics or {},
            tags=tags or [],
            description=description,
            file_path=str(registry_model_path),
            file_size=file_size,
            checksum=checksum,
            author=author,
            dependencies=dependencies or []
        )
        
        # Add to registry
        self.models[model_id] = metadata
        self._save_models()
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    def get_model_by_name_version(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata by name and version"""
        for model in self.models.values():
            if model.name == name and model.version == version:
                return model
        return None
    
    def list_models(self, name: str = None, tag: str = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        if name:
            models = [m for m in models if m.name == name]
            
        if tag:
            models = [m for m in models if tag in m.tags]
            
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry"""
        if model_id not in self.models:
            return False
            
        # Remove model file
        model_path = self.models[model_id].file_path
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Remove from registry
        del self.models[model_id]
        self._save_models()
        
        return True
    
    def get_latest_version(self, name: str) -> Optional[ModelMetadata]:
        """Get the latest version of a model by name"""
        models = [m for m in self.models.values() if m.name == name]
        if not models:
            return None
            
        # Sort by version (simple string sorting)
        models.sort(key=lambda x: x.version, reverse=True)
        return models[0]
    
    def export_registry(self, export_path: str):
        """Export registry to a file"""
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'models': {model_id: asdict(model) for model_id, model in self.models.items()}
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_registry(self, import_path: str):
        """Import registry from a file"""
        with open(import_path, 'r') as f:
            data = json.load(f)
            
        for model_id, model_data in data['models'].items():
            self.models[model_id] = ModelMetadata(**model_data)
            
        self._save_models()


class ModelComparison:
    """Model comparison utilities"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def compare_models(self, model_ids: List[str], 
                      metric: str = "accuracy") -> Dict[str, float]:
        """Compare models based on a specific metric"""
        results = {}
        for model_id in model_ids:
            model = self.registry.get_model(model_id)
            if model and metric in model.metrics:
                results[model_id] = model.metrics[metric]
            else:
                results[model_id] = 0.0
                
        return results
    
    def get_best_model(self, model_ids: List[str], 
                      metric: str = "accuracy", 
                      higher_is_better: bool = True) -> Optional[str]:
        """Get the best model based on a specific metric"""
        comparisons = self.compare_models(model_ids, metric)
        if not comparisons:
            return None
            
        if higher_is_better:
            return max(comparisons, key=comparisons.get)
        else:
            return min(comparisons, key=comparisons.get)


# Global registry instance
_model_registry = None


def get_model_registry(registry_path: str = "models/registry") -> ModelRegistry:
    """Get the global model registry instance"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry(registry_path)
    return _model_registry


def register_current_model(model_path: str, 
                          name: str, 
                          version: str, 
                          **kwargs) -> str:
    """Register the current model"""
    registry = get_model_registry()
    return registry.register_model(model_path, name, version, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create registry
    registry = ModelRegistry()
    
    # Example model registration (if file exists)
    # model_id = registry.register_model(
    #     model_path="demo_model.pt",
    #     name="sentiment_classifier",
    #     version="1.0.0",
    #     architecture="transformer",
    #     metrics={"accuracy": 0.85, "f1_score": 0.83},
    #     tags=["nlp", "sentiment", "demo"],
    #     description="Demo sentiment classifier model"
    # )
    # print(f"Registered model with ID: {model_id}")
    
    # List models
    models = registry.list_models()
    print(f"Registered models: {len(models)}")
    
    # Export registry
    registry.export_registry("model_registry_export.json")
    print("Registry exported to model_registry_export.json")