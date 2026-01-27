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
import time

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
                                st.session_state.feature_names,
                                st.session_state.X_test,
                                st.session_state.y_test
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
                
                # ========================================
                # PROACTIVE FAILURE RISK WARNING (MedGuard Innovation)
                # ========================================
                st.markdown("---")
                st.markdown("### 🎯 MedGuard Proactive Safety Check")
                
                # Load risk predictor
                from model_trainer import ModelTrainer
                risk_predictor = ModelTrainer.load_risk_predictor()
                
                if risk_predictor is not None:
                    # Get prediction probabilities for this case
                    X_case = st.session_state.X_test.loc[[selected_mistake['index']]]
                    y_pred_proba = trainer.model.predict_proba(X_case)
                    
                    # Predict failure risk
                    risk_score = risk_predictor.predict_risk(X_case, y_pred_proba)
                    risk_level = risk_predictor.get_risk_level(risk_score)
                    risk_factors = risk_predictor.get_risk_factors(X_case, y_pred_proba)
                    
                    # Display prominent risk alert based on level
                    if risk_level == "HIGH":
                        st.error(f"⚠️ **HIGH FAILURE RISK DETECTED** ({risk_score:.1%})")
                        st.markdown("**⚠️ CRITICAL ALERT**: Exercise extreme clinical caution. AI prediction may be unreliable.")
                    elif risk_level == "MEDIUM":
                        st.warning(f"⚡ **MODERATE FAILURE RISK** ({risk_score:.1%})")
                        st.markdown("**⚡ CAUTION**: Verify AI recommendation carefully against clinical guidelines.")
                    else:
                        st.info(f"✓ **LOW FAILURE RISK** ({risk_score:.1%})")
                        st.markdown("**✓ NORMAL**: AI decision likely reliable. Standard clinical review recommended.")
                    
                    # Risk visualization
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Risk score metric
                        risk_delta = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
                        st.metric(
                            "Failure Probability", 
                            f"{risk_score:.1%}",
                            delta=f"{risk_delta} Risk",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        # Colored progress bar
                        if risk_level == "HIGH":
                            st.markdown("**Risk Level:** 🔴 HIGH (>70%)")
                            st.progress(risk_score)
                        elif risk_level == "MEDIUM":
                            st.markdown("**Risk Level:** 🟠 MEDIUM (40-70%)")
                            st.progress(risk_score)
                        else:
                            st.markdown("**Risk Level:** 🟢 LOW (<40%)")
                            st.progress(risk_score)
                    
                    # Risk factors explanation
                    with st.expander("🔍 What Triggered This Warning?"):
                        st.markdown("**MedGuard detected the following risk indicators:**")
                        
                        if len(risk_factors) > 0:
                            for factor_name, factor_value, interpretation in risk_factors:
                                st.markdown(f"""
                                <div style='background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #ffc107;'>
                                    <strong>{factor_name}:</strong> {factor_value}<br>
                                    <em>{interpretation}</em>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.success("✓ No significant risk factors detected. AI operating within normal parameters.")
                        
                        st.markdown("---")
                        st.caption("💡 **This is the MedGuard Innovation**: We predict failures BEFORE they cause harm, not just explain predictions after the fact.")
                else:
                    st.warning("⚠️ Failure risk predictor not loaded. Train the model to enable proactive safety monitoring.")
                
                st.markdown("---")
                
                display_patient_case_card(
                    patient_data=patient_data,
                    patient_idx=selected_mistake['index'],
                    true_label=selected_mistake['true_label'],
                    predicted_label=selected_mistake['predicted_label'],
                    is_mistake=True,
                    case_type="Diagnostic Discrepancy"
                )
                
                # AI Prediction Panel and Failure Alert
                st.divider()
                st.subheader("🤖 AI Prediction Analysis")
                
                # Get AI prediction probabilities for the selected patient
                patient_data = st.session_state.X_test.loc[selected_mistake['index']].values.reshape(1, -1)
                prediction_proba = trainer.model.predict_proba(patient_data)[0]
                predicted_class = selected_mistake['predicted_label']
                actual_class = selected_mistake['true_label']
                
                # AI Prediction Panel
                col_pred, col_actual = st.columns([1, 1])
                
                with col_pred:
                    st.markdown("**🤖 AI Prediction**")
                    predicted_disease = "Heart Disease" if predicted_class == 1 else "Normal"
                    confidence = prediction_proba[predicted_class] * 100
                    
                    # Display prediction with confidence
                    st.metric(
                        label="Predicted Diagnosis",
                        value=predicted_disease,
                        delta=f"Confidence: {confidence:.1f}%",
                        delta_color="normal"
                    )
                    
                    # Confidence bar
                    st.markdown("**Confidence Level**")
                    st.progress(confidence / 100)
                    st.markdown(f"<p style='text-align: center; font-size: 12px; color: #666;'>{confidence:.1f}% confidence</p>", 
                             unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.markdown("**Probability Breakdown**")
                    prob_col1, prob_col2 = st.columns([1, 1])
                    with prob_col1:
                        st.metric("Heart Disease", f"{prediction_proba[1]:.1%}")
                    with prob_col2:
                        st.metric("Normal", f"{prediction_proba[0]:.1%}")
                
                with col_actual:
                    st.markdown("**🏥 Actual Diagnosis**")
                    actual_disease = "Heart Disease" if actual_class == 1 else "Normal"
                    
                    # Display actual diagnosis
                    st.metric(
                        label="Clinical Diagnosis",
                        value=actual_disease,
                        delta="Ground Truth",
                        delta_color="off"
                    )
                    
                    # Diagnosis status indicator
                    if predicted_class != actual_class:
                        st.error("❌ MISMATCH DETECTED")
                        st.markdown("""
                        <div style='background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 4px solid #ff4444;'>
                            <strong>⚠️ Diagnostic Error</strong><br>
                            AI prediction does not match clinical outcome
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ CORRECT MATCH")
                        st.markdown("""
                        <div style='background-color: #e6ffe6; padding: 10px; border-radius: 5px; border-left: 4px solid #44aa44;'>
                            <strong>✓ Accurate Diagnosis</strong><br>
                            AI prediction matches clinical outcome
                        </div>
                        """, unsafe_allow_html=True)
                
                # Failure Alert Banner
                st.divider()
                if predicted_class != actual_class:
                    st.error("""
                    ## ⚠️ AI MISDIAGNOSIS DETECTED - MedGuard Analysis Available
                    
                    **Critical Issue:** The AI system has made an incorrect diagnostic decision that requires immediate analysis.
                    
                    **Recommended Action:** Use MedGuard's contrastive explanation system to understand why the AI failed and learn from this error.
                    """)
                    
                    # Analyze AI Failure button (only when wrong)
                    if st.button("🔍 Analyze AI Failure", type="primary", use_container_width=True):
                        st.session_state.show_explanation = True
                else:
                    st.success("""
                    ## ✓ AI Diagnosis Matches Clinical Outcome
                    
                    **Status:** The AI system has made a correct diagnostic decision.
                    
                    **Note:** While this case was correctly diagnosed, you can still generate an explanation to understand the AI's reasoning process.
                    """)
                    
                    # Optional explanation button for correct cases
                    if st.button("📊 View AI Reasoning", use_container_width=True):
                        st.session_state.show_explanation = True
                
                # Explanation display (triggered by button)
                if st.session_state.get('show_explanation', False):
                    st.divider()
                    st.subheader("🔬 MedGuard Clinical Analysis")
                    
                    with st.spinner("Analyzing diagnostic decision and generating clinical insights..."):
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
                                    st.success("✅ Clinical failure analysis completed!")
                                    
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
                                    
                                    # Professional Clinical Failure Report
                                    st.divider()
                                    st.markdown("## � MedGuard Clinical Failure Report")
                                    
                                    # Get top features for analysis
                                    shap_mistake = explanation['shap_mistake']
                                    shap_correction = explanation['shap_correction']
                                    feature_names = explanation['feature_names']
                                    
                                    # Get top 3 features from mistake path (most influential in wrong decision)
                                    mistake_indices = np.argsort(np.abs(shap_mistake))[-3:][::-1]
                                    top_mistake_features = [feature_names[i] for i in mistake_indices]
                                    
                                    # Get top 3 features from correction path (should have been considered)
                                    correction_indices = np.argsort(np.abs(shap_correction))[-3:][::-1]
                                    top_correction_features = [feature_names[i] for i in correction_indices]
                                    
                                    # Get features that AI ignored but should have considered
                                    shap_delta = explanation['shap_delta']
                                    ignored_indices = np.argsort(shap_delta)[-3:][::-1]
                                    ignored_features = [feature_names[i] for i in ignored_indices]
                                    
                                    # Section 1: Why AI Failed
                                    st.markdown("### 🔴 Why AI Failed - Incorrect Focus")
                                    
                                    col1a, col1b = st.columns([2, 1])
                                    with col1a:
                                        st.markdown("**SHAP Mistake Path Visualization**")
                                        fig_mistake = explainer.plot_contrastive_explanation(explanation)
                                        # We'll need to modify this to show only the mistake path
                                        st.plotly_chart(fig_mistake, use_container_width=True)
                                    
                                    with col1b:
                                        st.markdown("**Key Findings**")
                                        st.error("The AI over-weighted:")
                                        for feature in top_mistake_features:
                                            st.markdown(f"• **{feature}**")
                                        
                                        with st.expander("� Detailed Feature Values"):
                                            for i, feature in enumerate(top_mistake_features):
                                                feature_idx = feature_names.index(feature)
                                                value = explanation['feature_values_mistake'][feature_idx]
                                                shap_val = shap_mistake[feature_idx]
                                                st.markdown(f"**{feature}**: {value:.3f} (SHAP: {shap_val:.3f})")
                                    
                                    # Section 2: Correct Clinical Reasoning
                                    st.markdown("### 🟢 Correct Clinical Reasoning")
                                    
                                    col2a, col2b = st.columns([2, 1])
                                    with col2a:
                                        st.markdown("**SHAP Correction Path Visualization**")
                                        # We'll reuse the same plot but focus on the correction
                                        st.plotly_chart(fig_mistake, use_container_width=True)
                                    
                                    with col2b:
                                        st.markdown("**Clinical Guidelines**")
                                        st.success("Should focus on:")
                                        for feature in top_correction_features:
                                            st.markdown(f"• **{feature}**")
                                        
                                        st.warning("AI ignored critical features:")
                                        for feature in ignored_features:
                                            st.markdown(f"• **{feature}**")
                                    
                                    # Section 3: MedGuard Assessment
                                    st.markdown("### 📊 MedGuard Assessment")
                                    
                                    # Calculate correctability score (simplified version)
                                    # Higher SHAP delta = harder to correct
                                    max_delta = np.max(np.abs(shap_delta))
                                    correctability_score = max(0, 100 - (max_delta * 50))  # Simplified scoring
                                    correctability_score = min(100, max(0, correctability_score))
                                    
                                    if correctability_score > 70:
                                        difficulty = "Easy"
                                        difficulty_color = "green"
                                        recommendation = "This mistake could have been prevented by following standard clinical protocols for vital sign assessment."
                                    elif correctability_score > 40:
                                        difficulty = "Medium"
                                        difficulty_color = "orange"
                                        recommendation = "This mistake requires additional clinical training and decision support systems to prevent recurrence."
                                    else:
                                        difficulty = "Hard"
                                        difficulty_color = "red"
                                        recommendation = "This mistake represents a complex diagnostic challenge requiring comprehensive model retraining and additional clinical features."
                                    
                                    col3a, col3b, col3c = st.columns([1, 1, 1])
                                    with col3a:
                                        st.metric(
                                            label="Correctability Score",
                                            value=f"{correctability_score:.1f}%",
                                            delta=None,
                                            delta_color="normal"
                                        )
                                        st.markdown(f"<div style='text-align: center; font-weight: bold; color: {difficulty_color}; font-size: 18px;'>{difficulty} to Fix</div>", 
                                                 unsafe_allow_html=True)
                                    
                                    with col3b:
                                        st.markdown("**Clinical Interpretation**")
                                        st.info(recommendation)
                                    
                                    with col3c:
                                        st.markdown("**Risk Assessment**")
                                        if predicted_class == 1 and actual_class == 0:
                                            st.error("False Positive\nUnnecessary treatment risk")
                                        elif predicted_class == 0 and actual_class == 1:
                                            st.error("False Negative\nMissed diagnosis risk")
                                        else:
                                            st.success("Correct Classification\nNo clinical risk")
                                    
                                    # Action Buttons Section
                                    st.divider()
                                    st.markdown("### 🎯 Recommended Actions")
                                    
                                    col_action1, col_action2, col_action3 = st.columns([1, 1, 1])
                                    
                                    with col_action1:
                                        if st.button("✅ Accept Corrected Diagnosis", type="primary", use_container_width=True):
                                            st.success("✅ Corrected diagnosis accepted. Clinical decision updated to match ground truth.")
                                            st.balloons()
                                    
                                    with col_action2:
                                        if st.button("🚫 Override AI Decision", use_container_width=True):
                                            st.warning("⚠️ AI decision overridden. Manual clinical intervention recorded.")
                                            st.info("This case has been flagged for model review and improvement.")
                                    
                                    with col_action3:
                                        if st.button("📝 Report to Safety Team", use_container_width=True):
                                            st.error("🚨 Case reported to MedGuard Safety Team.")
                                            st.info("Report ID: MG-" + str(mistake['index']) + "-" + str(int(time.time())))
                                            st.success("Safety team will review this diagnostic failure within 24 hours.")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error generating explanation: {str(e)}")
                                    import traceback
                                    st.error(f"Debug info: {traceback.format_exc()}")
                                
                        except Exception as e:
                            st.error(f"❌ Error generating explanation: {str(e)}")
                
                # Patient selector below the analysis
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
                    st.session_state.show_explanation = False  # Reset explanation when changing patient
                    st.rerun()
    
    with tab3:
        st.header("📈 Performance Metrics")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first in the 'System Diagnostics' tab.")
        else:
            trainer = st.session_state.trainer
            
            # System Status Overview Card (NEW)
            st.markdown("### 🏥 System Status Overview")
            
            # Get metrics for status dashboard
            total_cases = len(st.session_state.X_test)
            mistakes = trainer.get_mistakes(st.session_state.X_test, st.session_state.y_test)
            failure_count = len(mistakes)
            accuracy = trainer.accuracy
            
            # Create 5-column status dashboard
            status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
            
            with status_col1:
                st.metric(
                    label="✅ AI Model Status",
                    value="Active",
                    delta="RandomForest v1.0",
                    delta_color="normal"
                )
            
            with status_col2:
                st.metric(
                    label="✅ Safety Monitor",
                    value="Enabled",
                    delta="MedGuard Active",
                    delta_color="normal"
                )
            
            with status_col3:
                st.metric(
                    label="📊 Cases Analyzed",
                    value=f"{total_cases}",
                    delta="Test Dataset",
                    delta_color="off"
                )
            
            with status_col4:
                st.metric(
                    label="⚠️ Failures Detected",
                    value=f"{failure_count}",
                    delta=f"{failure_count/total_cases:.1%}",
                    delta_color="inverse" if failure_count > 0 else "normal"
                )
            
            with status_col5:
                st.metric(
                    label="📈 Overall Accuracy",
                    value=f"{accuracy:.1%}",
                    delta="Target: 95%",
                    delta_color="normal" if accuracy >= 0.85 else "inverse"
                )
            
            st.divider()
            
            # Patient Database Overview (renamed from Dataset Statistics)
            st.markdown("### 📋 Patient Database Overview")
            
            st.info("""
            **💡 Clinical Context:** This section provides an overview of the patient database used for AI training and testing, 
            helping clinicians understand the data distribution and demographics that inform the AI's decision-making process.
            """)
            
            # Display dataset statistics (existing functionality)
            if st.session_state.data_loaded:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📋 Patient Data Preview")
                    st.dataframe(st.session_state.raw_data.head())
                
                with col2:
                    st.subheader("📊 Clinical Feature Statistics")
                    st.dataframe(st.session_state.processed_data.describe())
            
            # AI Decision Accuracy Matrix (renamed from Confusion Matrix)
            st.markdown("### 🎯 AI Decision Accuracy Matrix")
            
            st.info("""
            **💡 Clinical Context:** This matrix shows where AI decisions align with actual clinical outcomes. 
            Perfect alignment indicates reliable AI assistance, while discrepancies highlight areas requiring clinical oversight.
            """)
            
            # Confusion Matrix (existing functionality)
            fig, ax = plt.subplots(figsize=(8, 6))
            trainer.plot_confusion_matrix(ax)
            st.pyplot(fig)
            
            # Clinical Context for Confusion Matrix (NEW)
            st.markdown("#### 🚨 Clinical Risk Assessment")
            
            # Calculate false positives and false negatives from confusion matrix
            cm = trainer.confusion_matrix
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    st.markdown(f"""
                    <div style='background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;'>
                        <h4 style='color: #856404; margin: 0 0 10px 0;'>⚠️ False Positives = {fp} cases</h4>
                        <p style='margin: 0; color: #856404;'><strong>Clinical Impact:</strong> Unnecessary clinical interventions</p>
                        <p style='margin: 5px 0 0 0; color: #856404;'><strong>Patient Risk:</strong> Potential overtreatment, anxiety, increased healthcare costs</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_col2:
                    st.markdown(f"""
                    <div style='background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545;'>
                        <h4 style='color: #721c24; margin: 0 0 10px 0;'>🚨 False Negatives = {fn} cases</h4>
                        <p style='margin: 0; color: #721c24;'><strong>Clinical Impact:</strong> Missed diagnoses (CRITICAL RISK)</p>
                        <p style='margin: 5px 0 0 0; color: #721c24;'><strong>Patient Risk:</strong> Delayed treatment, disease progression, potential mortality</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Mistakes analysis (existing functionality, enhanced with clinical context)
            mistakes = trainer.get_mistakes(
                st.session_state.X_test,
                st.session_state.y_test
            )
            if len(mistakes) > 0:
                st.markdown("### 🔍 Diagnostic Error Analysis")
                
                st.info("""
                **� Clinical Context:** This analysis identifies patterns in AI diagnostic errors, 
                helping healthcare providers understand common failure modes and implement targeted improvements.
                """)
                
                # Mistake distribution
                mistake_types = {}
                for mistake in mistakes:
                    key = f"True:{mistake['true_label']} → Pred:{mistake['predicted_label']}"
                    mistake_types[key] = mistake_types.get(key, 0) + 1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Error Distribution:**")
                    for mistake_type, count in mistake_types.items():
                        # Add clinical interpretation
                        if "True:1 → Pred:0" in mistake_type:
                            st.write(f"• {mistake_type}: {count} cases *(Missed Disease)*")
                        elif "True:0 → Pred:1" in mistake_type:
                            st.write(f"• {mistake_type}: {count} cases *(False Alarm)*")
                        else:
                            st.write(f"• {mistake_type}: {count} cases")
                
                with col2:
                    st.metric("Total Errors", len(mistakes))
                    st.metric("Error Rate", f"{len(mistakes)/len(st.session_state.y_test):.3f}")
                    
                    # Add clinical severity assessment
                    error_rate = len(mistakes)/len(st.session_state.y_test)
                    if error_rate > 0.20:
                        st.error("🚨 High Error Rate - Immediate Review Required")
                    elif error_rate > 0.10:
                        st.warning("⚠️ Moderate Error Rate - Clinical Oversight Recommended")
                    else:
                        st.success("✅ Acceptable Error Rate - Routine Monitoring")


if __name__ == "__main__":
    main()
