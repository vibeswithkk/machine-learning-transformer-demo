#!/usr/bin/env python3
"""
Demo script for the Machine Learning Transformer Demo

This script demonstrates:
1. Training a tiny model with dummy data
2. Evaluating the trained model
3. Information about running tests
"""

from src.train import train_model

def main():
    print("Machine Learning Transformer Demo")
    print("========================================")
    
    # Run the demo training
    print("Starting demo training...")
    model, trainer = train_model()
    
    # Evaluate the trained model
    print("\nEvaluating trained model...")
    metrics = trainer.evaluate()
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nDemo completed successfully!")
    
    # Provide information about testing
    print("\nTesting Information:")
    print("To run the comprehensive test suite with coverage reporting:")
    print("  pytest --cov=src tests/")
    print("")
    print("To generate an HTML coverage report:")
    print("  pytest --cov=src --cov-report=html tests/")
    print("")
    print("For more testing options, see TESTING.md")

if __name__ == "__main__":
    main()