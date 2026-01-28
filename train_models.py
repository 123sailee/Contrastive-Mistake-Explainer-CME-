"""
Quick training script to generate required models for expo
"""

import sys
import os
sys.path.append('src')

from data_loader import load_heart_disease
from model_trainer import ModelTrainer

def main():
    print("Training MedGuard AI models for expo...")
    
    # Load data
    print("Loading heart disease dataset...")
    X_train, X_test, y_train, y_test, feature_names, scaler, feature_descriptions = load_heart_disease()
    
    # Train primary model
    print("Training primary AI model...")
    trainer = ModelTrainer(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    trainer.train(
        X_train, y_train, 
        feature_names=feature_names,
        X_test=X_test, 
        y_test=y_test
    )
    
    # Save model
    print("Saving models...")
    trainer.save_model('models/primary_model.pkl')
    
    # Train risk predictor
    print("Training failure risk predictor...")
    trainer.train_failure_predictor(X_train, y_train, X_test, y_test)
    
    print("Training complete! Models ready for expo.")
    print(f"Models saved in: models/")
    print(f"Dataset size: {len(X_train)} training, {len(X_test)} test samples")

if __name__ == "__main__":
    main()
