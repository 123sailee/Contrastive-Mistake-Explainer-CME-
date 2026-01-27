# MedGuard AI - Clinical Decision Support System

**AI-Powered Clinical Decision Support & Failure Prevention**

*Built on Contrastive Mistake Explanation (CME) Technology*

## 🎯 Clinical Challenge

Traditional AI models in healthcare can make diagnostic errors that impact patient outcomes. MedGuard AI addresses this critical challenge by analyzing **why models make incorrect predictions** and providing **actionable insights for clinical decision support**:

1. **What factors led to the misdiagnosis?** (Error analysis)
2. **What SHOULD the model have considered?** (Correct decision path)
3. **How can we prevent similar errors?** (Clinical improvement insights)

## 💡 Clinical Innovation

**MedGuard AI** leverages **Contrastive Mistake Explanation (CME)** to provide **dual diagnostic insights** for each prediction error:

- 🔴 **Error Analysis**: SHAP explanation for the incorrect diagnosis
- 🟢 **Correct Decision Path**: SHAP explanation for a similar correct case
- 📊 **Clinical Comparison**: Side-by-side analysis showing diagnostic improvement opportunities

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/123sailee/Contrastive-Mistake-Explainer-CME-.git
cd Contrastive-Mistake-Explainer-CME-

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/streamlit_app.py
```

### Usage

1. **Load Patient Data**: Click "Load Patient Dataset" in the System Diagnostics tab
2. **Train Clinical Model**: Configure parameters in the sidebar and train the AI model
3. **Analyze Decisions**: Navigate to the "Clinical Decision Analysis" tab
4. **Review Diagnostic Insights**: Select cases to see side-by-side SHAP comparisons and clinical recommendations

## 📁 Project Structure

```
MedGuard-AI/
├── src/
│   ├── streamlit_app.py        # Main MedGuard AI application
│   ├── data_loader.py          # Patient data loading and preprocessing
│   ├── model_trainer.py        # Clinical model training & diagnostic analysis
│   └── cme_explainer.py        # SHAP-based clinical explanation engine
├── data/                       # Patient data directory (auto-created)
├── models/                     # Trained clinical models (auto-created)
├── requirements.txt
└── README.md
```

## 🔬 Methodology

### Phase 1 Implementation (Current)

1. **Dataset**: Heart Disease classification (UCI ML Repository)
2. **Model**: RandomForest classifier
3. **Explanation Method**: SHAP (TreeExplainer)
4. **Correction Path**: Nearest correctly-classified neighbor from same true class

### Algorithm

```
For each MISTAKE:
  1. Get SHAP explanation for the misclassified instance
  2. Find nearest correct example from same true class
  3. Get SHAP explanation for the correct example
  4. Generate contrastive visualization (side-by-side comparison)
  5. Compute SHAP delta to identify key differences
```

## 📊 Features

- **Interactive Training**: Adjust RandomForest hyperparameters in real-time
- **Model Evaluation**: Accuracy, precision, recall, F1, confusion matrix
- **Contrastive Explanations**: Side-by-side SHAP visualizations
- **Feature Comparison**: Detailed table showing SHAP values for both paths
- **Explanation Summary**: Auto-generated text describing what went wrong

## 🎓 Research Roadmap

### Phase 1 ✅ (Current)
- Basic contrastive explanations
- SHAP integration
- Nearest neighbor correction path

### Phase 2 (Future)
- **Correctability Scoring**: Rate mistakes as Easy/Medium/Hard to fix
- **Root Cause Analysis**: Automated diagnosis of mistake types

### Phase 3 (Future)
- **Meta-Model**: Predict when mistakes will occur
- **Proactive Warnings**: Alert users to high-risk predictions

### Phase 4 (Future)
- **Mistake Pattern Clustering**: Find systematic failure modes
- **Error Analytics Dashboard**: Comprehensive mistake analysis

### Phase 5 (Future)
- **Deployment**: Docker containerization
- **Hugging Face Spaces**: Public demo deployment

## 📚 Related Work

- **SHAP** (Lundberg & Lee, 2017): Feature importance explanations
- **LIME** (Ribeiro et al., 2016): Local interpretable model-agnostic explanations
- **Counterfactual Explanations** (Wachter et al., 2017): What inputs would change the prediction
- **Contrastive Explanations** (Miller, 2019): Human-centric explanation theory

## 🛠️ Technical Details

- **Language**: Python 3.10+
- **Framework**: Streamlit
- **ML**: scikit-learn (RandomForest)
- **XAI**: SHAP
- **Visualization**: Plotly, Matplotlib, Seaborn

## 📝 Citation

If you use this work, please cite:

```
Contrastive Mistake Explainer (CME): Learning What Models Should Have Seen
2026
GitHub: https://github.com/123sailee/Contrastive-Mistake-Explainer-CME-
```

## 🤝 Contributing

This is a research project. Contributions, suggestions, and feedback are welcome!

## 📄 License

MIT License - see LICENSE file for details

---

**Phase 1 Status**: ✅ Complete - Basic contrastive explanations with SHAP
