"""
CME Explainer - Contrastive Mistake Explanation Engine
Generates SHAP explanations for mistakes and their nearest correct neighbors.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CMEExplainer:
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize the CME explainer.
        
        Args:
            model: Trained sklearn model
            X_train: Training data for SHAP background
            feature_names: List of feature names
        """
        
        self.model = model
        self.feature_names = feature_names
        
        print("[INIT] Initializing SHAP TreeExplainer...")
        # Initialize SHAP explainer with a subset of training data (for speed)
        # Use 100 samples as background
        background_size = min(100, len(X_train))
        self.background_data = shap.sample(X_train, background_size)
        
        # Create TreeExplainer for RandomForest
        self.explainer = shap.TreeExplainer(self.model, self.background_data)
        print("[OK] SHAP explainer ready")
    
    def explain_instance(self, X_instance):
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            X_instance: Single instance to explain (1D or 2D array)
        
        Returns:
            SHAP values and base value
        """
        
        # Ensure 2D shape
        if hasattr(X_instance, 'values'):
            X_np = X_instance.values
        else:
            X_np = X_instance
            
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        
        # Get SHAP values with additivity check disabled
        shap_values = self.explainer.shap_values(X_np, check_additivity=False)
        
        # For binary classification, SHAP returns values for both classes
        # We want the values for the predicted class (class 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (disease present)
        
        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        return shap_values[0], base_value
    
    def generate_contrastive_explanation(self, X_mistake, X_correction, 
                                          feature_values_mistake, feature_values_correction):
        """
        Generate contrastive explanation comparing mistake path vs correction path.
        
        Args:
            X_mistake: Misclassified instance
            X_correction: Nearest correct example
            feature_values_mistake: Original feature values (before scaling)
            feature_values_correction: Original feature values for correction
        
        Returns:
            Dictionary with contrastive explanation data
        """
        
        # Get SHAP explanations for both
        shap_mistake, base_mistake = self.explain_instance(X_mistake)
        shap_correction, base_correction = self.explain_instance(X_correction)
        
        # Compute delta (difference in feature importance)
        shap_delta = shap_correction - shap_mistake
        
        # Ensure flattened arrays to prevent shape errors
        shap_mistake = np.array(shap_mistake).flatten()
        shap_correction = np.array(shap_correction).flatten()
        shap_delta = np.array(shap_delta).flatten()
        
        # Create explanation dictionary
        explanation = {
            'shap_mistake': shap_mistake,
            'shap_correction': shap_correction,
            'shap_delta': shap_delta,
            'base_value': base_mistake,
            'feature_names': self.feature_names,
            'feature_values_mistake': feature_values_mistake,
            'feature_values_correction': feature_values_correction
        }
        
        # Identify key differences
        explanation['top_overweighted'] = self._get_top_features(shap_mistake, n=3, sign='positive')
        explanation['top_underweighted'] = self._get_top_features(shap_delta, n=3, sign='positive')
        explanation['top_missing'] = self._get_top_features(-shap_delta, n=3, sign='positive')
        
        return explanation
    
    def _get_top_features(self, shap_values, n=5, sign='positive'):
        """
        Get top N features by absolute SHAP value.

        Args:
            shap_values: SHAP values array
            n: Number of top features
            sign: 'positive', 'negative', or 'both'

        Returns:
            List of (feature_name, shap_value) tuples
        """
        # Ensure 1D array
        shap_values = np.array(shap_values).flatten()

        # Validate dimensions
        if len(shap_values) != len(self.feature_names):
            print(f"[WARNING] SHAP values length ({len(shap_values)}) != feature_names length ({len(self.feature_names)})")
            # Truncate to minimum length
            min_len = min(len(shap_values), len(self.feature_names))
            shap_values = shap_values[:min_len]

        # Ensure n doesn't exceed array length
        n = min(n, len(shap_values))

        if sign == 'positive':
            indices = np.argsort(shap_values)[-n:][::-1]
        elif sign == 'negative':
            indices = np.argsort(shap_values)[:n]
        else:  # both
            indices = np.argsort(np.abs(shap_values))[-n:][::-1]

        # Double-check indices are valid and convert properly
        top_features = []
        for i in indices:
            idx = int(i)
            if 0 <= idx < len(self.feature_names) and 0 <= idx < len(shap_values):
                top_features.append((self.feature_names[idx], float(shap_values[idx])))

        return top_features

    def plot_contrastive_explanation(self, explanation):
        """
        Create interactive Plotly visualization for contrastive explanation.

        Args:
            explanation: Dictionary from generate_contrastive_explanation

        Returns:
            Plotly figure object
        """

        shap_mistake = explanation['shap_mistake']
        shap_correction = explanation['shap_correction']
        feature_names = explanation['feature_names']
        feature_values_mistake = explanation['feature_values_mistake']
        feature_values_correction = explanation['feature_values_correction']

        # Sort features by absolute difference (most different first)
        abs_diff = np.abs(shap_correction - shap_mistake)
        
        # Ensure we only use valid indices (min length of all arrays)
        min_length = min(len(feature_names), len(shap_mistake), len(shap_correction), len(feature_values_mistake), len(feature_values_correction))
        
        # Only consider valid indices
        valid_indices = np.arange(min_length)
        valid_abs_diff = abs_diff[:min_length]
        
        sorted_indices = valid_indices[np.argsort(valid_abs_diff)[::-1]]
        
        # Take top 5 most different features (simplified for speed)
        top_n = min(5, min_length)
        top_indices = sorted_indices[:top_n]

        # Create subplots: side by side
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Mistake Path (Wrong Reasoning)', 'Correction Path (Correct Reasoning)'),
            horizontal_spacing=0.15
        )

        # Mistake path (LEFT)
        y_pos = np.arange(top_n)
        colors_mistake = ['red' if float(shap_mistake[i]) > 0 else 'darkred' for i in top_indices]

        fig.add_trace(
            go.Bar(
                y=[feature_names[i] for i in top_indices],
                x=shap_mistake[top_indices],
                orientation='h',
                marker=dict(color=colors_mistake),
                text=[f"{feature_names[i]}={feature_values_mistake[i]:.2f}" for i in top_indices],
                textposition='outside',
                name='Mistake',
                hovertemplate='<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Correction path (RIGHT)
        colors_correction = ['green' if float(shap_correction[i]) > 0 else 'darkgreen' for i in top_indices]

        fig.add_trace(
            go.Bar(
                y=[feature_names[i] for i in top_indices],
                x=shap_correction[top_indices],
                orientation='h',
                marker=dict(color=colors_correction),
                text=[f"{feature_names[i]}={feature_values_correction[i]:.2f}" for i in top_indices],
                textposition='outside',
                name='Correction',
                hovertemplate='<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="<- Decreases Risk | Increases Risk ->", row=1, col=1)
        fig.update_xaxes(title_text="<- Decreases Risk | Increases Risk ->", row=1, col=2)

        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="Contrastive Explanation: What Went Wrong vs What Should Have Happened",
            title_font_size=16
        )

        return fig

    def get_explanation_summary(self, explanation):
        """
        Generate human-readable text summary of the contrastive explanation.

        Args:
            explanation: Dictionary from generate_contrastive_explanation

        Returns:
            String with explanation summary
        """

        summary_parts = []

        # Over-weighted features
        if explanation['top_overweighted']:
            overweighted = explanation['top_overweighted']
            summary_parts.append("**Model Over-relied On:**")
            for feat, val in overweighted:
                if val > 0:
                    summary_parts.append(f"- {feat} (SHAP: {val:.3f})")

        # Under-weighted features
        if explanation['top_underweighted']:
            underweighted = explanation['top_underweighted']
            summary_parts.append("\n**Model Should Have Focused More On:**")
            for feat, val in underweighted:
                if val > 0:
                    summary_parts.append(f"- {feat} (SHAP delta: +{val:.3f})")

        # Missing features
        if explanation['top_missing']:
            missing = explanation['top_missing']
            summary_parts.append("\n**Model Ignored Important Features:**")
            for feat, val in missing:
                if val > 0:
                    summary_parts.append(f"- {feat} (SHAP delta: -{val:.3f})")

        return "\n".join(summary_parts)

    def precompute_shap_cache(self, X_test, save_path='models/shap_cache.pkl', batch_size=10):
        """
        Pre-compute SHAP values for all test samples and cache to disk.
        Uses batch processing to show progress and avoid memory issues.

        Args:
            X_test: Test dataset
            save_path: Path to save cache file
            batch_size: Number of samples to process per batch

        Returns:
            Dictionary with cached SHAP values
        """
        import os
        import joblib

        n_samples = len(X_test)
        print(f"\n{'='*60}")
        print(f"[START] Pre-computing SHAP values for {n_samples} test samples")
        print(f"[INFO] Batch size: {batch_size} samples")
        print(f"{'='*60}\n")

        # Convert to numpy if needed
        X_np = X_test.values if hasattr(X_test, 'values') else X_test

        # Process in batches
        all_shap_values = []
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = X_np[start_idx:end_idx]

            print(f"[PROCESSING] Processing batch {batch_idx + 1}/{n_batches} (samples {start_idx + 1}-{end_idx})...")

            # Compute SHAP for this batch (with additivity check disabled)
            batch_shap = self.explainer.shap_values(batch_data, check_additivity=False)

            # For binary classification, extract class 1 values
            if isinstance(batch_shap, list):
                batch_shap = batch_shap[1]

            all_shap_values.append(batch_shap)
            print(f"[OK] Batch {batch_idx + 1}/{n_batches} complete ({end_idx}/{n_samples} total samples)")

        # Combine all batches
        print(f"\n[COMBINING] Combining {n_batches} batches...")
        shap_values = np.vstack(all_shap_values)
        print(f"[OK] Combined shape: {shap_values.shape}")

        # Cache structure
        cache = {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            'feature_names': self.feature_names,
            'indices': X_test.index.tolist() if hasattr(X_test, 'index') else list(range(len(X_test)))
        }

        # Save to disk
        print(f"\n[SAVING] Saving cache to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(cache, save_path)

        print(f"[OK] SHAP cache saved successfully!")
        print(f"{'='*60}\n")

        return cache
