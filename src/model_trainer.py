"""
Model Trainer for CME Project
Implements RandomForest classifier with evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os


class ModelTrainer:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum number of samples required to split an internal node
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the RandomForest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
        """
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = feature_names if feature_names else list(range(X_train.shape[1]))
        
        # Get training metrics
        y_pred_train = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        return {
            'train_accuracy': train_accuracy,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def get_mistakes(self, X_test, y_test):
        """
        Get all misclassified instances from test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            DataFrame with mistake information
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before getting mistakes")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Find mistakes
        mistake_indices = np.where(y_pred != y_test)[0]
        
        mistakes = []
        for idx in mistake_indices:
            mistake_info = {
                'index': X_test.index[idx] if hasattr(X_test, 'index') else idx,
                'true_label': y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx],
                'predicted_label': y_pred[idx],
                'confidence': np.max(y_pred_proba[idx]),
                'predicted_proba': y_pred_proba[idx].tolist()
            }
            mistakes.append(mistake_info)
        
        return mistakes
    
    def predict(self, X, return_proba=True):
        """
        Make predictions on new data.
        
        Args:
            X: Features
            return_proba: Whether to return probabilities
        
        Returns:
            Predictions and optionally probabilities
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        y_pred = self.model.predict(X)
        
        if return_proba:
            y_pred_proba = self.model.predict_proba(X)
            return y_pred, y_pred_proba
        
        return y_pred
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = True
    
    def plot_confusion_matrix(self, ax):
        """
        Plot confusion matrix on given matplotlib axis.
        
        Args:
            ax: Matplotlib axis object to plot on
        """
        if not hasattr(self, 'confusion_matrix') or self.confusion_matrix is None:
            raise ValueError("Model must be evaluated first to get confusion matrix")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create heatmap
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   square=True, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')


def find_nearest_correct_example(X_train, y_train, X_mistake, y_true, model, n_neighbors=1):
    """
    Find the nearest correctly-classified training example from the same true class.
    This serves as the "correction path" for contrastive explanation.
    
    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training labels
        X_mistake: The misclassified instance
        y_true: The true label of the mistake
        model: Trained model
        n_neighbors: Number of nearest neighbors to return
    
    Returns:
        Index of nearest correct example(s)
    """
    
    # Convert to numpy arrays if needed
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_mistake_np = X_mistake.values if hasattr(X_mistake, 'values') else X_mistake
    
    # Reshape mistake if needed
    if X_mistake_np.ndim == 1:
        X_mistake_np = X_mistake_np.reshape(1, -1)
    
    # Get predictions for training set
    y_train_pred = model.predict(X_train_np)
    
    # Filter to:
    # 1. Same true class as the mistake
    # 2. Correctly classified by the model
    same_class_mask = (y_train == y_true)
    correctly_classified_mask = (y_train_pred == y_train)
    valid_mask = same_class_mask & correctly_classified_mask
    
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        # Fallback: just find same class examples (even if misclassified)
        valid_indices = np.where(same_class_mask)[0]
        
        if len(valid_indices) == 0:
            # Ultimate fallback: return first training example
            return [0]
    
    # Calculate distances to all valid examples
    valid_X = X_train_np[valid_indices]
    distances = np.linalg.norm(valid_X - X_mistake_np, axis=1)
    
    # Get indices of nearest neighbors
    nearest_local_indices = np.argsort(distances)[:n_neighbors]
    nearest_global_indices = valid_indices[nearest_local_indices]
    
    return nearest_global_indices.tolist()
