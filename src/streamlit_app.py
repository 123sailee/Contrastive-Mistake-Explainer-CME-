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


def display_patient_case_card(patient_data, patient_idx, true_label, predicted_label, is_mistake=True, case_type="Diagnostic Discrepancy"):
    """
    Display patient information as a professional medical case card.
    
    Args:
        patient_data: Patient feature data (pandas Series)
        patient_idx: Patient ID/index
        true_label: Actual diagnosis
        predicted_label: AI prediction
        is_mistake: Whether this is a misclassified case
        case_type: Type of case being displayed
    """
    
    # Determine border color based on classification
    border_color = "#ff4444" if is_mistake else "#4444ff"  # Red for mistakes, Blue for correct
    
    # Custom CSS for card styling
    card_css = f"""
    <style>
    .patient-card-{patient_idx} {{
        border: 3px solid {border_color};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .patient-header-{patient_idx} {{
        background-color: {border_color};
        color: white;
        padding: 10px;
        border-radius: 7px 7px 0 0;
        margin: -20px -20px 15px -20px;
        font-weight: bold;
        text-align: center;
    }}
    .vital-metric {{
        display: inline-block;
        margin: 5px 10px;
        padding: 5px 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        font-size: 14px;
    }}
    .clinical-flag {{
        display: inline-block;
        margin: 3px 5px;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: bold;
    }}
    .flag-normal {{ background-color: #d4edda; color: #155724; }}
    .flag-abnormal {{ background-color: #f8d7da; color: #721c24; }}
    .flag-warning {{ background-color: #fff3cd; color: #856404; }}
    </style>
    """
    
    st.markdown(card_css, unsafe_allow_html=True)
    
    with st.container():
        # Patient Header
        diagnosis_status = "❌ INCORRECT" if is_mistake else "✅ CORRECT"
        st.markdown(f"""
        <div class="patient-header-{patient_idx}">
            <h3 style="margin: 0; font-size: 18px;">PATIENT CASE #{patient_idx} - {case_type}</h3>
            <p style="margin: 5px 0 0 0; font-size: 14px;">AI Diagnosis Status: {diagnosis_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main patient information
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("**📋 Patient Information**")
            st.markdown(f"**ID:** #{patient_idx}")
            st.markdown(f"**Age:** {int(patient_data['age'])} years")
            
            # Sex interpretation
            sex_text = "Male" if patient_data['sex'] == 1 else "Female"
            st.markdown(f"**Sex:** {sex_text}")
            
            # Chest pain type interpretation
            cp_types = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
            cp_type = cp_types[int(patient_data['cp'])] if 0 <= patient_data['cp'] <= 3 else f"Type {int(patient_data['cp'])}"
            st.markdown(f"**Chest Pain:** {cp_type}")
        
        with col2:
            st.markdown("**🏥 Key Vitals & Measurements**")
            
            # Create vital metrics display
            vitals_html = f"""
            <div class="vital-metric">🩸 **BP:** {int(patient_data['trestbps'])} mmHg</div>
            <div class="vital-metric">🧪 **Cholesterol:** {int(patient_data['chol'])} mg/dL</div>
            <div class="vital-metric">❤️ **Max HR:** {int(patient_data['thalach'])} bpm</div>
            <div class="vital-metric">📉 **ST Depression:** {patient_data['oldpeak']:.1f} mm</div>
            """
            st.markdown(vitals_html, unsafe_allow_html=True)
        
        with col3:
            st.markdown("**🚨 Clinical Flags**")
            
            # Fasting Blood Sugar
            fbs_status = "Elevated" if patient_data['fbs'] == 1 else "Normal"
            fbs_class = "flag-abnormal" if patient_data['fbs'] == 1 else "flag-normal"
            st.markdown(f'<span class="clinical-flag {fbs_class}">🩸 FBS: {fbs_status}</span>', unsafe_allow_html=True)
            
            # Resting ECG
            ecg_types = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
            ecg_type = ecg_types[int(patient_data['restecg'])] if 0 <= patient_data['restecg'] <= 2 else f"Type {int(patient_data['restecg'])}"
            ecg_class = "flag-normal" if patient_data['restecg'] == 0 else "flag-warning"
            st.markdown(f'<span class="clinical-flag {ecg_class}">📊 ECG: {ecg_type}</span>', unsafe_allow_html=True)
            
            # Exercise Angina
            exang_status = "Present" if patient_data['exang'] == 1 else "Absent"
            exang_class = "flag-abnormal" if patient_data['exang'] == 1 else "flag-normal"
            st.markdown(f'<span class="clinical-flag {exang_class}">🏃 Exercise Angina: {exang_status}</span>', unsafe_allow_html=True)
        
        # Diagnosis Comparison
        st.divider()
        diag_col1, diag_col2, diag_col3 = st.columns([1, 1, 1])
        
        with diag_col1:
            actual_disease = "Heart Disease" if true_label == 1 else "Normal"
            actual_color = "#ff4444" if true_label == 1 else "#44aa44"
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                <h4 style="margin: 0; color: #333;">ACTUAL DIAGNOSIS</h4>
                <p style="margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: {actual_color};">
                    {actual_disease}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with diag_col2:
            predicted_disease = "Heart Disease" if predicted_label == 1 else "Normal"
            predicted_color = "#ff4444" if predicted_label == 1 else "#44aa44"
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                <h4 style="margin: 0; color: #333;">AI PREDICTION</h4>
                <p style="margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: {predicted_color};">
                    {predicted_disease}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with diag_col3:
            if is_mistake:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #ffe6e6; border-radius: 5px; border: 2px solid #ff4444;">
                    <h4 style="margin: 0; color: #d00;">⚠️ DIAGNOSTIC ERROR</h4>
                    <p style="margin: 5px 0 0 0; font-size: 14px; font-weight: bold; color: #d00;">
                        Requires Analysis
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background-color: #e6ffe6; border-radius: 5px; border: 2px solid #44aa44;">
                    <h4 style="margin: 0; color: #080;">✅ CORRECT DIAGNOSIS</h4>
                    <p style="margin: 5px 0 0 0; font-size: 14px; font-weight: bold; color: #080;">
                        Reference Case
                    </p>
                </div>
                """, unsafe_allow_html=True)


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
                
                # Initialize selected mistake if not in session state
                if 'selected_mistake_idx' not in st.session_state:
                    st.session_state.selected_mistake_idx = 0
                
                # Display patient case card for selected mistake
                selected_mistake = mistakes[st.session_state.selected_mistake_idx]
                patient_data = st.session_state.X_test.loc[selected_mistake['index']]
                
                display_patient_case_card(
                    patient_data=patient_data,
                    patient_idx=selected_mistake['index'],
                    true_label=selected_mistake['true_label'],
                    predicted_label=selected_mistake['predicted_label'],
                    is_mistake=True,
                    case_type="Diagnostic Discrepancy"
                )
                
                # Patient selector below the card
                st.divider()
                st.subheader("📋 Select Patient Case")
                
                mistake_options = [f"Patient {i} (Actual: {m['true_label']}, Predicted: {m['predicted_label']})" 
                                 for i, m in enumerate(mistakes)]
                
                new_selection = st.selectbox(
                    "Choose a diagnostic discrepancy to analyze:",
                    range(len(mistake_options)),
                    index=st.session_state.selected_mistake_idx,
                    format_func=lambda x: mistake_options[x]
                )
                
                # Update session state if selection changed
                if new_selection != st.session_state.selected_mistake_idx:
                    st.session_state.selected_mistake_idx = new_selection
                    st.rerun()
                
                if st.button("🔬 Generate Clinical Explanation", type="primary"):
                    with st.spinner("Analyzing diagnostic discrepancy and finding correct decision path..."):
                        try:
                            mistake = mistakes[st.session_state.selected_mistake_idx]
                            
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
                                    
                                    # Display both patient cases side by side
                                    st.subheader("🏥 Patient Case Comparison")
                                    col_case1, col_case2 = st.columns([1, 1])
                                    
                                    with col_case1:
                                        st.markdown("**❌ Incorrect Diagnosis Case**")
                                        display_patient_case_card(
                                            patient_data=st.session_state.X_test.loc[mistake['index']],
                                            patient_idx=mistake['index'],
                                            true_label=mistake['true_label'],
                                            predicted_label=mistake['predicted_label'],
                                            is_mistake=True,
                                            case_type="Diagnostic Error"
                                        )
                                    
                                    with col_case2:
                                        st.markdown("**✅ Correct Reference Case**")
                                        correction_data = st.session_state.X_train.iloc[correction['index']]
                                        # Get the model's prediction for the correction case
                                        correction_pred = trainer.model.predict(correction_data.values.reshape(1, -1))[0]
                                        
                                        display_patient_case_card(
                                            patient_data=correction_data,
                                            patient_idx=correction['index'],
                                            true_label=correction['true_label'],
                                            predicted_label=correction_pred,
                                            is_mistake=False,
                                            case_type="Correct Diagnosis"
                                        )
                                    
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
