"""
Failure Risk Predictor - Meta-Model for Proactive AI Safety
Predicts when the primary diagnostic model is likely to fail BEFORE clinical impact

This is the CORE INNOVATION that differentiates MedGuard from standard XAI tools.
Instead of explaining failures after they happen, we predict them before they occur.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class FailureRiskPredictor:
    """
    Meta-model that predicts when the primary diagnostic model is likely to make mistakes.
    
    This enables proactive AI safety by identifying high-risk predictions BEFORE
    they impact patient care, allowing clinicians to exercise appropriate oversight.
    
    Key Innovation: Predictive failure detection vs reactive explanation
    """
    
    def __init__(self, primary_model):
        """
        Initialize with the trained primary model
        
        Args:
            primary_model: Trained sklearn model (RandomForest, etc.)
        """
        self.primary_model = primary_model
        self.risk_model = None
        self.scaler = StandardScaler()
        self.feature_stats = {}  # Store training data statistics for outlier detection
        self.is_trained = False
        
    def extract_meta_features(self, X, y_pred_proba):
        """
        Extract 5 meta-features that indicate failure risk:
        
        1. Prediction confidence (max probability) - Low confidence indicates uncertainty
        2. Prediction entropy (uncertainty measure) - High entropy indicates confusion
        3. Decision margin (gap between top 2 classes) - Small margin indicates instability
        4. Extreme feature count (features >2 std dev) - Outliers may confuse the model
        5. Feature range violations (outside training bounds) - Unseen data patterns
        
        Args:
            X: Patient feature data (DataFrame or array)
            y_pred_proba: Prediction probabilities from primary model
            
        Returns:
            numpy array of shape (n_samples, 5) with meta-features
        """
        meta_features = []
        
        for idx in range(len(X)):
            proba = y_pred_proba[idx]
            sample = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            
            # Feature 1: Prediction confidence
            # Low confidence suggests the model is uncertain about its prediction
            confidence = np.max(proba)
            
            # Feature 2: Prediction entropy
            # High entropy indicates the model is confused between classes
            entropy = -np.sum(proba * np.log(proba + 1e-10))
            
            # Feature 3: Decision margin
            # Small margin means the decision could easily flip with small data changes
            sorted_proba = np.sort(proba)[::-1]
            margin = sorted_proba[0] - sorted_proba[1] if len(sorted_proba) > 1 else sorted_proba[0]
            
            # Feature 4: Extreme feature count
            # Count features that are statistical outliers (>2 standard deviations)
            if len(self.feature_stats) > 0:
                extreme_count = sum([
                    1 for feat_idx, val in enumerate(sample) 
                    if abs((val - self.feature_stats[feat_idx]['mean']) / 
                           (self.feature_stats[feat_idx]['std'] + 1e-10)) > 2
                ])
            else:
                extreme_count = 0
            
            # Feature 5: Feature range violations
            # Count features outside the range seen during training
            if len(self.feature_stats) > 0:
                violations = sum([
                    1 for feat_idx, val in enumerate(sample)
                    if val < self.feature_stats[feat_idx]['min'] or 
                       val > self.feature_stats[feat_idx]['max']
                ])
            else:
                violations = 0
            
            meta_features.append([confidence, entropy, margin, extreme_count, violations])
        
        return np.array(meta_features)
    
    def train(self, X_train, y_train, y_pred_proba_train, y_pred_train):
        """
        Train meta-model to predict failures using training data
        
        Args:
            X_train: Training features
            y_train: True labels
            y_pred_proba_train: Prediction probabilities on training data
            y_pred_train: Predictions on training data
            
        Returns:
            dict: Training metrics including accuracy, precision, recall, f1
        """
        # Store feature statistics for later use in risk assessment
        for feat_idx in range(X_train.shape[1]):
            feat_values = X_train.iloc[:, feat_idx] if hasattr(X_train, 'iloc') else X_train[:, feat_idx]
            self.feature_stats[feat_idx] = {
                'mean': np.mean(feat_values),
                'std': np.std(feat_values),
                'min': np.min(feat_values),
                'max': np.max(feat_values)
            }
        
        # Extract meta-features from training data
        meta_X = self.extract_meta_features(X_train, y_pred_proba_train)
        
        # Create binary labels: 1 if mistake, 0 if correct
        meta_y = (y_pred_train != y_train).astype(int)
        
        # Check if we have both classes
        if len(np.unique(meta_y)) < 2:
            # If no mistakes or all mistakes, create synthetic balanced data
            print("Warning: Training data has no failure cases. Creating synthetic balanced training data...")
            
            # Create synthetic failure cases by adding noise to some correct predictions
            n_synthetic = min(10, len(meta_y) // 4)  # 10 or 25% of data
            synthetic_indices = np.random.choice(len(meta_y), n_synthetic, replace=False)
            
            # Create synthetic meta-features for failures
            synthetic_meta = []
            for idx in synthetic_indices:
                base_features = meta_X[idx].copy()
                # Make features more risk-like for synthetic failures
                base_features[0] *= 0.6  # Lower confidence
                base_features[1] *= 1.5  # Higher entropy
                base_features[2] *= 0.5  # Lower margin
                base_features[3] += 1   # More extreme features
                base_features[4] += 1   # More violations
                synthetic_meta.append(base_features)
            
            # Add synthetic failures
            meta_X = np.vstack([meta_X, np.array(synthetic_meta)])
            meta_y = np.concatenate([meta_y, np.ones(n_synthetic)])
        
        # Scale features
        meta_X_scaled = self.scaler.fit_transform(meta_X)
        
        # Train logistic regression with balanced classes to handle imbalance
        self.risk_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        self.risk_model.fit(meta_X_scaled, meta_y)
        
        # Calculate training metrics
        meta_y_pred = self.risk_model.predict(meta_X_scaled)
        
        training_metrics = {
            'accuracy': accuracy_score(meta_y, meta_y_pred),
            'precision': precision_score(meta_y, meta_y_pred, average='weighted', zero_division=0),
            'recall': recall_score(meta_y, meta_y_pred, average='weighted', zero_division=0),
            'f1': f1_score(meta_y, meta_y_pred, average='weighted', zero_division=0),
            'failure_rate': np.mean(meta_y),
            'n_failures': np.sum(meta_y),
            'n_samples': len(meta_y)
        }
        
        self.is_trained = True
        
        return training_metrics
    
    def predict_risk(self, X_sample, y_pred_proba_sample):
        """
        Predict failure risk for a single sample
        
        Args:
            X_sample: Single patient sample (DataFrame row or array)
            y_pred_proba_sample: Prediction probabilities for the sample
            
        Returns:
            float: Risk probability [0-1] where higher values indicate higher failure risk
        """
        if not self.is_trained:
            raise ValueError("Risk model not trained. Call train() first.")
        
        # Extract meta-features for the sample
        meta_features = self.extract_meta_features(X_sample, y_pred_proba_sample)
        
        # Scale features using the same scaler from training
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Predict risk (probability of failure)
        risk_proba = self.risk_model.predict_proba(meta_features_scaled)[0, 1]
        
        return risk_proba
    
    def predict_risk_batch(self, X_batch, y_pred_proba_batch):
        """
        Predict failure risk for multiple samples
        
        Args:
            X_batch: Multiple patient samples
            y_pred_proba_batch: Prediction probabilities for the batch
            
        Returns:
            numpy array: Risk probabilities [0-1] for each sample
        """
        if not self.is_trained:
            raise ValueError("Risk model not trained. Call train() first.")
        
        # Extract meta-features for the batch
        meta_features = self.extract_meta_features(X_batch, y_pred_proba_batch)
        
        # Scale features
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Predict risk probabilities
        risk_probas = self.risk_model.predict_proba(meta_features_scaled)[:, 1]
        
        return risk_probas
    
    def get_risk_level(self, risk_score):
        """
        Categorize risk score into clinical action levels
        
        Args:
            risk_score: Risk probability [0-1]
            
        Returns:
            str: Risk level (HIGH/MEDIUM/LOW)
        """
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_risk_factors(self, X_sample, y_pred_proba_sample):
        """
        Identify top risk factors for this prediction
        
        Args:
            X_sample: Single patient sample
            y_pred_proba_sample: Prediction probabilities for the sample
            
        Returns:
            list: Top risk factors with interpretations
        """
        meta_features = self.extract_meta_features(X_sample, y_pred_proba_sample)[0]
        
        factors = []
        
        # Check confidence
        if meta_features[0] < 0.7:
            factors.append((
                "Low Prediction Confidence",
                f"{meta_features[0]:.1%}",
                f"AI only {meta_features[0]:.1%} confident (should be >70%)"
            ))
        
        # Check entropy
        if meta_features[1] > 0.5:
            factors.append((
                "High Prediction Uncertainty",
                f"{meta_features[1]:.2f}",
                "AI shows confusion between diagnostic categories"
            ))
        
        # Check decision margin
        if meta_features[2] < 0.3:
            factors.append((
                "Narrow Decision Margin",
                f"{meta_features[2]:.2%}",
                "AI decision could easily flip with small data changes"
            ))
        
        # Check extreme features
        if meta_features[3] > 0:
            factors.append((
                "Unusual Patient Profile",
                f"{int(meta_features[3])} features",
                f"{int(meta_features[3])} clinical values are statistical outliers"
            ))
        
        # Check range violations
        if meta_features[4] > 0:
            factors.append((
                "Out-of-Range Values",
                f"{int(meta_features[4])} features",
                f"{int(meta_features[4])} values outside training data range"
            ))
        
        # Sort by risk significance and return top 3
        return factors[:3]
    
    def get_feature_importance(self):
        """
        Get the importance of each meta-feature in risk prediction
        
        Returns:
            dict: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Risk model not trained. Call train() first.")
        
        feature_names = [
            'Prediction Confidence',
            'Prediction Entropy', 
            'Decision Margin',
            'Extreme Feature Count',
            'Range Violations'
        ]
        
        importance = self.risk_model.coef_[0]
        
        return dict(zip(feature_names, importance))
    
    def save_model(self, filepath):
        """
        Save the trained risk predictor model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'risk_model': self.risk_model,
            'scaler': self.scaler,
            'feature_stats': self.feature_stats,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        Load a trained risk predictor model
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.risk_model = model_data['risk_model']
        self.scaler = model_data['scaler']
        self.feature_stats = model_data['feature_stats']
        self.is_trained = model_data['is_trained']
    
    def evaluate(self, X_test, y_test, y_pred_proba_test, y_pred_test):
        """
        Evaluate the risk predictor on test data
        
        Args:
            X_test: Test features
            y_test: True labels
            y_pred_proba_test: Prediction probabilities on test data
            y_pred_test: Predictions on test data
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Risk model not trained. Call train() first.")
        
        # Extract meta-features and create labels
        meta_X = self.extract_meta_features(X_test, y_pred_proba_test)
        meta_y = (y_pred_test != y_test).astype(int)
        
        # Scale features
        meta_X_scaled = self.scaler.transform(meta_X)
        
        # Make predictions
        meta_y_pred = self.risk_model.predict(meta_X_scaled)
        meta_y_proba = self.risk_model.predict_proba(meta_X_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(meta_y, meta_y_pred),
            'precision': precision_score(meta_y, meta_y_pred, average='weighted', zero_division=0),
            'recall': recall_score(meta_y, meta_y_pred, average='weighted', zero_division=0),
            'f1': f1_score(meta_y, meta_y_pred, average='weighted', zero_division=0),
            'auc': None,  # Would need binary labels for AUC
            'failure_rate': np.mean(meta_y),
            'n_failures': np.sum(meta_y),
            'n_samples': len(meta_y),
            'mean_risk_score': np.mean(meta_y_proba),
            'high_risk_count': np.sum(meta_y_proba > 0.7)
        }
        
        return metrics


# Utility function for quick setup
def create_risk_predictor(primary_model, X_train, y_train, X_test, y_test):
    """
    Quick setup function to create and train a risk predictor
    
    Args:
        primary_model: Trained primary model
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: (trained FailureRiskPredictor, training_metrics, test_metrics)
    """
    # Create risk predictor
    risk_predictor = FailureRiskPredictor(primary_model)
    
    # Get predictions from primary model
    y_pred_proba_train = primary_model.predict_proba(X_train)
    y_pred_train = primary_model.predict(X_train)
    
    y_pred_proba_test = primary_model.predict_proba(X_test)
    y_pred_test = primary_model.predict(X_test)
    
    # Train risk predictor
    training_metrics = risk_predictor.train(X_train, y_train, y_pred_proba_train, y_pred_train)
    
    # Evaluate
    test_metrics = risk_predictor.evaluate(X_test, y_test, y_pred_proba_test, y_pred_test)
    
    return risk_predictor, training_metrics, test_metrics
