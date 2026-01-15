"""
Correctability Scorer for CME Project
Quantifies how "fixable" each mistake is based on explanation similarity.
"""

import numpy as np
from scipy.spatial.distance import cosine


class CorrectabilityScorer:
    def __init__(self):
        """Initialize the correctability scorer."""
        # Weights for scoring components (can be tuned)
        self.alpha = 0.4  # Feature coverage weight
        self.beta = 0.3   # Confidence gap weight
        self.gamma = 0.3  # Delta magnitude weight
    
    def calculate_score(self, explanation, confidence_wrong, confidence_correct=None):
        """
        Calculate correctability score for a mistake.
        
        Score ranges from 0-100:
        - 70-100: EASY FIX - Just needs feature reweighting
        - 30-70: MEDIUM FIX - Partial feature blindness
        - 0-30: HARD FIX - Fundamental reasoning error
        
        Args:
            explanation: Dictionary from CMEExplainer.generate_contrastive_explanation
            confidence_wrong: Model confidence in wrong prediction
            confidence_correct: Model confidence for correct class (optional)
        
        Returns:
            Dictionary with score, category, and components
        """
        
        shap_mistake = explanation['shap_mistake']
        shap_correction = explanation['shap_correction']
        shap_delta = explanation['shap_delta']
        
        # Component 1: Feature Coverage
        # How many important features did the model consider (even if weighted wrong)?
        # High coverage = Easy Fix (just reweight)
        # Low coverage = Hard Fix (model blind to features)
        feature_coverage = self._calculate_feature_coverage(shap_mistake, shap_correction)
        
        # Component 2: Confidence Gap
        # How confident was the model in its wrong answer?
        # Low confidence = Easy Fix (model was uncertain)
        # High confidence = Hard Fix (model very sure of wrong answer)
        confidence_gap = 1.0 - confidence_wrong  # Invert so higher is better
        
        # Component 3: Delta Magnitude  
        # How different are the explanations?
        # Small delta = Easy Fix (similar reasoning)
        # Large delta = Hard Fix (completely different reasoning needed)
        delta_magnitude = 1.0 / (1.0 + np.linalg.norm(shap_delta))  # Normalize and invert
        
        # Combine components
        raw_score = (
            self.alpha * feature_coverage +
            self.beta * confidence_gap +
            self.gamma * delta_magnitude
        )
        
        # Scale to 0-100
        score = raw_score * 100
        
        # Categorize
        category = self._categorize_score(score)
        
        return {
            'score': score,
            'category': category,
            'components': {
                'feature_coverage': feature_coverage,
                'confidence_gap': confidence_gap,
                'delta_magnitude': delta_magnitude
            },
            'interpretation': self._get_interpretation(score, category)
        }
    
    def _calculate_feature_coverage(self, shap_mistake, shap_correction):
        """
        Calculate what proportion of important features were considered.
        
        Compare which features have non-zero SHAP in both explanations.
        """
        
        # Get absolute SHAP values
        abs_mistake = np.abs(shap_mistake)
        abs_correction = np.abs(shap_correction)
        
        # Define "important" as top 50% by magnitude in correction path
        correction_threshold = np.median(abs_correction)
        important_features = abs_correction > correction_threshold
        
        # How many important features did the mistake path consider?
        mistake_threshold = np.median(abs_mistake)
        considered_features = abs_mistake > mistake_threshold
        
        # Calculate overlap
        overlap = np.sum(important_features & considered_features)
        total_important = np.sum(important_features)
        
        if total_important == 0:
            return 0.5  # Default if no important features
        
        coverage = overlap / total_important
        return coverage
    
    def _categorize_score(self, score):
        """
        Categorize the correctability score.
        
        Args:
            score: Score from 0-100
        
        Returns:
            Category string
        """
        
        if score >= 70:
            return "EASY FIX"
        elif score >= 30:
            return "MEDIUM FIX"
        else:
            return "HARD FIX"
    
    def _get_interpretation(self, score, category):
        """
        Get human-readable interpretation of the score.
        
        Args:
            score: Correctability score
            category: Category (EASY/MEDIUM/HARD FIX)
        
        Returns:
            Interpretation string
        """
        
        interpretations = {
            "EASY FIX": "The model has the right features but wrong weightings. Simple retraining with more diverse examples could fix this.",
            "MEDIUM FIX": "The model is partially blind to important features. May need feature engineering or architectural changes.",
            "HARD FIX": "Fundamental reasoning error. The model's learned patterns are very different from the correct reasoning. Requires significant intervention."
        }
        
        return interpretations.get(category, "Unknown category")
    
    def calculate_cosine_similarity(self, shap_mistake, shap_correction):
        """
        Calculate cosine similarity between mistake and correction explanations.
        
        High similarity = explanations are similar (Easy Fix)
        Low similarity = explanations are very different (Hard Fix)
        
        Returns:
            Similarity score from -1 to 1
        """
        
        # Avoid division by zero
        if np.all(shap_mistake == 0) or np.all(shap_correction == 0):
            return 0.0
        
        similarity = 1 - cosine(shap_mistake, shap_correction)
        return similarity
