"""
Data Loader for Heart Disease Dataset
Handles downloading, preprocessing, and splitting data for the CME project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_heart_disease(test_size=0.3, random_state=42):
    """
    Load and preprocess the Heart Disease dataset from UCI repository.
    
    Args:
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
        feature_names: List of feature names
        scaler: Fitted StandardScaler object
    """
    
    # Heart Disease dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Feature descriptions for UI
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1=male, 0=female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1=yes, 0=no)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)'
    }
    
    try:
        # Load dataset
        df = pd.read_csv(url, names=column_names, na_values='?')
        
        # Handle missing values (replace with median)
        df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != 'object' else x)
        
        # Convert target to binary (0 = no disease, 1 = disease)
        # Original: 0 = no disease, 1-4 = different severity levels
        df['target'] = (df['target'] > 0).astype(int)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        feature_names = list(X.columns)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, feature_descriptions
        
    except Exception as e:
        raise Exception(f"Error loading Heart Disease dataset: {str(e)}")


def load_custom_dataset(file_path, target_column, test_size=0.3, random_state=42):
    """
    Load a custom CSV dataset.
    
    Args:
        file_path: Path to CSV file
        target_column: Name of the target column
        test_size: Proportion of data for testing
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
        feature_names: List of feature names
        scaler: Fitted StandardScaler object
    """
    
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Convert categorical columns to numeric using one-hot encoding
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        feature_names = list(X.columns)
        feature_descriptions = {col: col for col in feature_names}
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, feature_descriptions
        
    except Exception as e:
        raise Exception(f"Error loading custom dataset: {str(e)}")


def get_dataset_statistics(X_train, X_test, y_train, y_test):
    """
    Get statistics about the dataset for display.
    
    Returns:
        Dictionary with dataset statistics
    """
    
    stats = {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'n_classes': len(np.unique(y_train)),
        'class_distribution_train': y_train.value_counts().to_dict(),
        'class_distribution_test': y_test.value_counts().to_dict(),
        'feature_names': list(X_train.columns)
    }
    
    return stats
