# Contrastive Mistake Explainer (CME)

**Learning What Models Should Have Seen - A Novel Approach to AI Explainability**

## 🎯 Research Problem

Traditional explainable AI (XAI) methods focus on explaining **why models make predictions**. But when a model makes a **mistake**, we need to understand:

1. **What did the model focus on?** (Mistake path)
2. **What SHOULD it have focused on?** (Correction path)
3. **What's the difference?** (Contrastive delta)

## 💡 Novel Contribution

**Contrastive Mistake Explanation (CME)** generates **dual explanations** for each mistake:

- 🔴 **Mistake Path**: SHAP explanation for the wrong prediction
- 🟢 **Correction Path**: SHAP explanation for a correct similar example
- 📊 **Contrastive View**: Side-by-side comparison showing the "explanation gap"

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd C:\Users\saile\Desktop\AICORRECTOR

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/streamlit_app.py
```

### Usage

1. **Load Dataset**: Click "Load Heart Disease Dataset" in the Data & Training tab
2. **Train Model**: Configure parameters in the sidebar and train the RandomForest model
3. **Explore Mistakes**: Navigate to the "Mistake Analysis" tab
4. **View Contrastive Explanations**: Select a mistake to see side-by-side SHAP comparisons

## 📁 Project Structure

```
AICORRECTOR/
├── src/
│   ├── streamlit_app.py        # Main Streamlit application
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── model_trainer.py        # RandomForest training & nearest neighbor finder
│   └── cme_explainer.py        # SHAP-based contrastive explanation engine
├── data/                       # Data directory (auto-created)
├── models/                     # Saved models (auto-created)
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
GitHub: [Repository URL]
```

## 🤝 Contributing

This is a research project. Contributions, suggestions, and feedback are welcome!

## 📄 License

[Your License Here]

---

**Phase 1 Status**: ✅ Complete - Basic contrastive explanations with SHAP
