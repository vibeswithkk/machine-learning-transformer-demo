#!/usr/bin/env python3
"""
Extended example for sentiment analysis with real datasets
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model import TransformerClassifier, ModelConfig
from src.train import train_model, TrainingConfig
from src.inference import InferenceEngine


def create_sample_dataset() -> Tuple[List[str], List[int]]:
    """Create a sample dataset for demonstration"""
    # Positive examples
    positive_texts = [
        "I love this product! It's amazing and works perfectly.",
        "Great service and fast delivery. Highly recommended!",
        "Excellent quality and value for money. Very satisfied.",
        "Outstanding performance and user-friendly interface.",
        "This exceeded my expectations. Fantastic product!",
        "Wonderful experience. Will definitely buy again.",
        "Impressive features and reliable performance.",
        "Top-notch quality and excellent customer support.",
        "I'm thrilled with this purchase. Great value!",
        "Superb product that delivers on its promises."
    ]
    
    # Negative examples
    negative_texts = [
        "Terrible product. Complete waste of money.",
        "Poor quality and bad customer service. Avoid!",
        "Disappointing experience. Not worth the price.",
        "Faulty product with many issues. Very frustrated.",
        "Awful service and slow delivery. Never again!",
        "Substandard quality and misleading description.",
        "Broken on arrival and difficult return process.",
        "Overpriced for the quality provided. Unhappy.",
        "Worst purchase ever. Completely useless.",
        "Horrible experience. Would not recommend to anyone."
    ]
    
    # Neutral examples
    neutral_texts = [
        "The product is okay. Nothing special but does the job.",
        "Average quality and standard service. As expected.",
        "It's fine, not great but not terrible either.",
        "Decent product for the price point.",
        "Meets basic requirements but lacks advanced features.",
        "Standard performance. No major complaints or praises.",
        "Functional but could be improved in several areas.",
        "Acceptable quality for occasional use.",
        "Moderate satisfaction. Room for improvement.",
        "Satisfactory but not impressive."
    ]
    
    # Combine all texts and create labels
    texts = positive_texts + negative_texts + neutral_texts
    labels = [0] * len(positive_texts) + [1] * len(negative_texts) + [2] * len(neutral_texts)
    
    return texts, labels


def prepare_data_for_training() -> Tuple[List[str], List[int], List[str], List[int]]:
    """Prepare data for training and evaluation"""
    # Create sample dataset
    texts, labels = create_sample_dataset()
    
    # Split into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return train_texts, train_labels, test_texts, test_labels


def train_sentiment_model():
    """Train a sentiment analysis model"""
    print("Preparing data for training...")
    train_texts, train_labels, test_texts, test_labels = prepare_data_for_training()
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Configure model
    model_config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        max_position_embeddings=128,
        num_classes=3,  # positive, negative, neutral
        dropout=0.1
    )
    
    # Configure training
    training_config = TrainingConfig(
        output_dir="./sentiment-results",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-4,
        logging_steps=5,
        eval_steps=10,
        save_steps=10,
        gradient_checkpointing=True
    )
    
    print("Starting model training...")
    model, trainer = train_model(
        model_config=model_config,
        training_config=training_config,
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=test_texts,
        eval_labels=test_labels
    )
    
    # Save the trained model
    model_path = "sentiment_model.pt"
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
    
    return model, model_path


def evaluate_model(model_path: str, test_texts: List[str], test_labels: List[int]):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # Create inference engine
    engine = InferenceEngine(model_path=model_path)
    
    # Make predictions
    predictions = []
    for text in test_texts:
        result = engine.predict_single(text)
        # Convert predicted class to integer (assuming classes are "0", "1", "2")
        pred_class = int(result.predicted_class)
        predictions.append(pred_class)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, 
                              target_names=["Positive", "Negative", "Neutral"]))


def demo_inference(model_path: str):
    """Demonstrate inference with the trained model"""
    print("\nDemonstrating inference...")
    
    # Create inference engine
    engine = InferenceEngine(model_path=model_path)
    
    # Test texts
    test_examples = [
        "This product is absolutely fantastic! I love it.",
        "Terrible quality and poor service. Very disappointed.",
        "It's an okay product, does what it's supposed to do.",
        "Amazing features and excellent value for money!",
        "Not satisfied with this purchase. Would not recommend."
    ]
    
    print("Predictions:")
    for text in test_examples:
        result = engine.predict_single(text)
        class_names = ["Positive", "Negative", "Neutral"]
        predicted_name = class_names[int(result.predicted_class)]
        
        print(f"\nText: {text}")
        print(f"Predicted Class: {predicted_name}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")


def main():
    """Main function"""
    print("Sentiment Analysis Demo")
    print("=" * 30)
    
    # Prepare data
    _, _, test_texts, test_labels = prepare_data_for_training()
    
    # Check if model exists
    model_path = "sentiment_model.pt"
    if not os.path.exists(model_path):
        print("Training sentiment analysis model...")
        model, model_path = train_sentiment_model()
    else:
        print("Using existing model...")
    
    # Evaluate model
    evaluate_model(model_path, test_texts, test_labels)
    
    # Demonstrate inference
    demo_inference(model_path)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()