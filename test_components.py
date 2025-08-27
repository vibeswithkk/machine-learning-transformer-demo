#!/usr/bin/env python3
"""
Test script to verify that all new components work correctly
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all new components can be imported"""
    components = [
        'api',
        'preprocessing',
        'monitoring',
        'registry'
    ]
    
    failed_imports = []
    
    for component in components:
        try:
            __import__(f'src.{component}')
            print(f"✓ {component} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {component}: {e}")
            failed_imports.append(component)
    
    return len(failed_imports) == 0

def test_preprocessing():
    """Test the preprocessing module"""
    try:
        from src.preprocessing import AdvancedTextPreprocessor
        
        # Create preprocessor
        preprocessor = AdvancedTextPreprocessor()
        
        # Test text normalization
        text = "This is a sample text with àccénts and punctuation!"
        normalized = preprocessor.normalize_text(text)
        print(f"✓ Text normalization: {normalized}")
        
        # Test tokenization
        tokens = preprocessor.tokenize_text(normalized)
        print(f"✓ Text tokenization: {len(tokens)} tokens")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False

def test_monitoring():
    """Test the monitoring module"""
    try:
        from src.monitoring import ModelMonitor, PredictionMetrics
        
        # Create monitor
        monitor = ModelMonitor("test_model")
        
        # Create test metrics
        metrics = PredictionMetrics(
            timestamp=1234567890.0,
            processing_time_ms=50.0,
            text_length=100,
            model_version="1.0.0",
            predicted_class="positive",
            confidence=0.95,
            cache_hit=False
        )
        
        # Record metrics
        monitor.record_prediction(metrics)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"✓ Monitoring test passed: {summary['total_predictions']} predictions recorded")
        
        return True
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
        return False

def test_registry():
    """Test the registry module"""
    try:
        from src.registry import ModelRegistry
        
        # Create registry
        registry = ModelRegistry("test_registry")
        
        # List models (should be empty)
        models = registry.list_models()
        print(f"✓ Registry test passed: {len(models)} models in registry")
        
        return True
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Testing new components...")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_imports),
        ("Preprocessing Tests", test_preprocessing),
        ("Monitoring Tests", test_monitoring),
        ("Registry Tests", test_registry)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"  Result: PASSED")
        else:
            print(f"  Result: FAILED")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! 🎉")
        return True
    else:
        print("Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)