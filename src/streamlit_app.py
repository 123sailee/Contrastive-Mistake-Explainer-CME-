"""
CME Explainer - Contrastive Mistake Explanation Engine
Generates SHAP explanations for mistakes and their nearest correct neighbors.
"""

import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_heart_disease
from model_trainer import ModelTrainer, find_nearest_correct_example
from cme_explainer import CMEExplainer


# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="MedGuard AI - Clinical Decision Support",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 MedGuard AI")
    st.markdown("*AI-Powered Clinical Decision Support & Failure Prevention*")
    
    # Sidebar for model configuration
    st.sidebar.header("⚙️ Model Configuration")
    
    # Model hyperparameters
    n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
    max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=20, value=10, step=1)
    min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=20, value=5, step=1)
    random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42, step=1)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 System Diagnostics", "🔍 Clinical Decision Analysis", "📈 Performance Metrics"])
    
    with tab1:
        st.header("📊 System Diagnostics")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🚀 Load Patient Dataset", type="primary"):
                with st.spinner("Loading and preprocessing data..."):
                    try:
                        # Load and preprocess data
                        X_train, X_test, y_train, y_test, feature_names, scaler, feature_descriptions = load_heart_disease()
                        
                        # Create a processed dataframe for display
                        processed_data = pd.concat([X_train, X_test])
                        raw_data = processed_data.copy()
                        
                        st.session_state.raw_data = raw_data
                        st.session_state.processed_data = processed_data
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = feature_names
                        st.session_state.scaler = scaler
                        st.session_state.feature_descriptions = feature_descriptions
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Patient dataset loaded successfully!")
                        st.info(f"📊 Dataset shape: {processed_data.shape}")
                        st.info(f"🎯 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading data: {str(e)}")
        
        with col2:
            if st.session_state.data_loaded:
                st.subheader("📋 Patient Data Preview")
                st.dataframe(st.session_state.raw_data.head())
                
                st.subheader("📊 Clinical Feature Statistics")
                st.dataframe(st.session_state.processed_data.describe())
        
        # Model Training Section
        if st.session_state.data_loaded:
            st.divider()
            st.subheader("🤖 Model Training")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("� Train Clinical Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            # Initialize trainer
                            trainer = ModelTrainer(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=random_state
                            )
                            
                            # Train model
                            trainer.train(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.feature_names
                            )
                            
                            # Evaluate model
                            eval_results = trainer.evaluate(
                                st.session_state.X_test,
                                st.session_state.y_test
                            )
                            
                            # Store metrics in trainer for easy access
                            trainer.accuracy = eval_results['accuracy']
                            trainer.precision = eval_results['precision']
                            trainer.recall = eval_results['recall']
                            trainer.f1 = eval_results['f1']
                            trainer.y_pred = eval_results['y_pred']
                            trainer.y_pred_proba = eval_results['y_pred_proba']
                            trainer.confusion_matrix = eval_results['confusion_matrix']
                            
                            # Initialize explainer
                            explainer = CMEExplainer(
                                trainer.model,
                                st.session_state.X_train,
                                st.session_state.feature_names
                            )
                            
                            st.session_state.trainer = trainer
                            st.session_state.explainer = explainer
                            st.session_state.model_trained = True
                            
                            st.success("✅ Clinical model trained successfully!")
                            st.info(f"🎯 Model Accuracy: {trainer.accuracy:.3f}")
                            
                        except Exception as e:
                            st.error(f"❌ Error training model: {str(e)}")
            
            with col2:
                if st.session_state.model_trained:
                    st.subheader("📈 Diagnostic Results")
                    trainer = st.session_state.trainer
                    
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.metric("Accuracy", f"{trainer.accuracy:.3f}")
                        st.metric("Precision", f"{trainer.precision:.3f}")
                    with col2b:
                        st.metric("Recall", f"{trainer.recall:.3f}")
                        st.metric("F1-Score", f"{trainer.f1:.3f}")
    
    with tab2:
        st.header("🔍 Clinical Decision Analysis")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first in the 'System Diagnostics' tab.")
        else:
            trainer = st.session_state.trainer
            explainer = st.session_state.explainer
            
            # Get mistakes
            mistakes = trainer.get_mistakes(
                st.session_state.X_test,
                st.session_state.y_test
            )
            
            if len(mistakes) == 0:
                st.success("🎉 No mistakes found! The model achieved perfect accuracy on the test set.")
            else:
                st.info(f"🔍 Found {len(mistakes)} diagnostic discrepancies")
                
                # Mistake selector
                mistake_options = [f"Patient {i} (Actual: {m['true_label']}, Predicted: {m['predicted_label']})" 
                                 for i, m in enumerate(mistakes)]
                
                selected_mistake_idx = st.selectbox(
                    "Select a diagnostic discrepancy to analyze:",
                    range(len(mistake_options)),
                    format_func=lambda x: mistake_options[x]
                )
                
                if st.button("🔬 Generate Clinical Explanation", type="primary"):
                    with st.spinner("Analyzing diagnostic discrepancy and finding correct decision path..."):
                        try:
                            mistake = mistakes[selected_mistake_idx]
                            
                            # Get the mistake instance first
                            X_mistake = st.session_state.X_test.loc[mistake['index']].values.reshape(1, -1)
                            
                            # Get correction
                            correction_indices = find_nearest_correct_example(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                X_mistake,
                                mistake['true_label'],
                                trainer.model
                            )
                            
                            if len(correction_indices) == 0:
                                correction = None
                            else:
                                try:
                                    correction_idx = correction_indices[0]
                                    
                                    # Validate the correction index
                                    if correction_idx < 0 or correction_idx >= len(st.session_state.X_train):
                                        st.error(f"❌ Invalid correction index: {correction_idx}. Training data length: {len(st.session_state.X_train)}")
                                        correction = None
                                    else:
                                        correction = {
                                            'index': correction_idx,
                                            'true_label': st.session_state.y_train.iloc[correction_idx] if hasattr(st.session_state.y_train, 'iloc') else st.session_state.y_train[correction_idx]
                                        }
                                except Exception as e:
                                    st.error(f"❌ Error accessing correction data: {str(e)}")
                                    correction = None
                            
                            if correction is None:
                                st.error("❌ Could not find a suitable correct diagnostic example.")
                            else:
                                try:
                                    # Generate contrastive explanation
                                    st.info(f"🔍 Using correct diagnostic example at index {correction['index']}")
                                    X_correction = st.session_state.X_train.iloc[correction['index']].values.reshape(1, -1)
                                    
                                    # Get original feature values (use scaled values since we don't have original)
                                    feature_vals_mistake = X_mistake[0]
                                    feature_vals_correction = X_correction[0]
                                    
                                    explanation = explainer.generate_contrastive_explanation(
                                        X_mistake, X_correction,
                                        feature_vals_mistake, feature_vals_correction
                                    )
                                    
                                    # Display results
                                    st.success("✅ Clinical explanation generated!")
                                    
                                    # Summary section
                                    st.subheader("📝 Clinical Analysis Summary")
                                    summary = explainer.get_explanation_summary(explanation)
                                    st.markdown(summary)
                                    
                                    # Visualization
                                    st.subheader("📊 Diagnostic Comparison Visualization")
                                    fig = explainer.plot_contrastive_explanation(explanation)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Feature comparison table
                                    st.subheader("🔍 Clinical Feature Comparison")
                                    comparison_data = []
                                    for i, feat_name in enumerate(explanation['feature_names']):
                                        comparison_data.append({
                                            'Feature': feat_name,
                                            'Incorrect Value': f"{explanation['feature_values_mistake'][i]:.3f}",
                                            'Correct Value': f"{explanation['feature_values_correction'][i]:.3f}",
                                            'Incorrect SHAP': f"{explanation['shap_mistake'][i]:.3f}",
                                            'Correct SHAP': f"{explanation['shap_correction'][i]:.3f}",
                                            'SHAP Difference': f"{explanation['shap_delta'][i]:.3f}"
                                        })
                                    
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"❌ Error generating explanation: {str(e)}")
                                    import traceback
                                    st.error(f"Debug info: {traceback.format_exc()}")
                                
                        except Exception as e:
                            st.error(f"❌ Error generating explanation: {str(e)}")
    
    with tab3:
        st.header("📈 Performance Metrics")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first in the 'System Diagnostics' tab.")
        else:
            trainer = st.session_state.trainer
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{trainer.accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{trainer.precision:.3f}")
            with col3:
                st.metric("Recall", f"{trainer.recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{trainer.f1:.3f}")
            
            # Confusion Matrix
            st.subheader("🔢 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            trainer.plot_confusion_matrix(ax)
            st.pyplot(fig)
            
            # Mistakes analysis
            mistakes = trainer.get_mistakes(
                st.session_state.X_test,
                st.session_state.y_test
            )
            if len(mistakes) > 0:
                st.subheader("🔍 Mistake Analysis")
                
                # Mistake distribution
                mistake_types = {}
                for mistake in mistakes:
                    key = f"True:{mistake['true_label']} → Pred:{mistake['predicted_label']}"
                    mistake_types[key] = mistake_types.get(key, 0) + 1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Mistake Distribution:**")
                    for mistake_type, count in mistake_types.items():
                        st.write(f"• {mistake_type}: {count} cases")
                
                with col2:
                    st.metric("Total Mistakes", len(mistakes))
                    st.metric("Mistake Rate", f"{len(mistakes)/len(st.session_state.y_test):.3f}")


if __name__ == "__main__":
    main()
