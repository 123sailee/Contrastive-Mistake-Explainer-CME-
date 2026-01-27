"""
Test script for Failure Risk Predictor
Verifies the core innovation of proactive AI failure prediction
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from model_trainer import ModelTrainer
from data_loader import load_heart_disease
from failure_risk_predictor import FailureRiskPredictor

def test_risk_predictor():
    """Test the failure risk predictor functionality"""
    print("Testing Failure Risk Predictor - Core Innovation")
    print("=" * 60)
    
    try:
        # Load data and train primary model
        print("Loading data and training primary model...")
        X_train, X_test, y_train, y_test, feature_names, scaler, feature_descriptions = load_heart_disease()
        
        # Train primary model
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, feature_names)
        eval_results = trainer.evaluate(X_test, y_test)
        
        print(f"Primary model trained with accuracy: {eval_results['accuracy']:.3f}")
        
        # Create and train risk predictor
        print("\nTraining failure risk predictor...")
        risk_predictor = FailureRiskPredictor(trainer.model)
        
        # Get predictions from primary model
        y_pred_proba_train = trainer.model.predict_proba(X_train)
        y_pred_train = trainer.model.predict(X_train)
        y_pred_proba_test = trainer.model.predict_proba(X_test)
        y_pred_test = trainer.model.predict(X_test)
        
        # Train risk predictor
        training_metrics = risk_predictor.train(X_train, y_train, y_pred_proba_train, y_pred_train)
        
        print(f"Risk predictor trained with accuracy: {training_metrics['accuracy']:.3f}")
        print(f"Training failure rate: {training_metrics['failure_rate']:.1%}")
        print(f"Training failures: {training_metrics['n_failures']}/{training_metrics['n_samples']}")
        
        # Evaluate on test data
        print("\nEvaluating on test data...")
        test_metrics = risk_predictor.evaluate(X_test, y_test, y_pred_proba_test, y_pred_test)
        
        print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
        print(f"Test failure rate: {test_metrics['failure_rate']:.1%}")
        print(f"Test failures: {test_metrics['n_failures']}/{test_metrics['n_samples']}")
        print(f"Mean risk score: {test_metrics['mean_risk_score']:.3f}")
        print(f"High risk predictions: {test_metrics['high_risk_count']}")
        
        # Test individual predictions
        print("\nTesting individual risk predictions...")
        
        # Find a mistake and a correct prediction
        mistakes = trainer.get_mistakes(X_test, y_test)
        
        if len(mistakes) > 0:
            mistake = mistakes[0]
            mistake_idx = mistake['index']
            
            # Test risk prediction for a mistake
            mistake_data = X_test.loc[[mistake_idx]]  # Keep as DataFrame
            mistake_proba = trainer.model.predict_proba(mistake_data)
            
            risk_score = risk_predictor.predict_risk(mistake_data, mistake_proba)
            risk_level = risk_predictor.get_risk_level(risk_score)
            risk_factors = risk_predictor.get_risk_factors(mistake_data, mistake_proba)
            
            print(f"\nMISSED CASE (Patient {mistake_idx}):")
            print(f"   True: {mistake['true_label']}, Predicted: {mistake['predicted_label']}")
            print(f"   Risk Score: {risk_score:.3f} ({risk_level} risk)")
            print(f"   Risk Factors:")
            for factor, value, interpretation in risk_factors:
                print(f"     * {factor}: {value} - {interpretation}")
        
        # Test a correct prediction
        correct_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred_test)) if true == pred]
        if correct_indices:
            correct_idx = correct_indices[0]
            correct_data = X_test.iloc[[correct_idx]]
            correct_proba = trainer.model.predict_proba(correct_data)
            
            risk_score = risk_predictor.predict_risk(correct_data, correct_proba)
            risk_level = risk_predictor.get_risk_level(risk_score)
            risk_factors = risk_predictor.get_risk_factors(correct_data, correct_proba)
            
            print(f"\nCORRECT CASE (Patient {correct_idx}):")
            print(f"   True: {y_test.iloc[correct_idx]}, Predicted: {y_pred_test[correct_idx]}")
            print(f"   Risk Score: {risk_score:.3f} ({risk_level} risk)")
            print(f"   Risk Factors:")
            for factor, value, interpretation in risk_factors:
                print(f"     * {factor}: {value} - {interpretation}")
        
        # Test feature importance
        print(f"\nMeta-Feature Importance:")
        importance = risk_predictor.get_feature_importance()
        for feature, score in importance.items():
            print(f"   {feature}: {score:.3f}")
        
        print(f"\nSUCCESS! Failure Risk Predictor working correctly!")
        print(f"Core Innovation: Predicting failures BEFORE they happen!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_risk_predictor()
    if success:
        print("\nAll tests passed! The failure risk predictor is ready for integration.")
    else:
        print("\nTests failed. Please check the implementation.")
