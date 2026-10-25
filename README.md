# Rotary Pounder Regression & Textural Degradation Framework

This repository hosts the data preprocessing pipeline, ensemble machine-learning models, and SHAP interpretability scripts accompanying the research study: **"Predicting Mechanical Resistance and Textural Degradation in Rotary Yam Pounders Using Ensemble Machine Learning"**.

## 📌 Abstract / Overview
Mechanized rotary pounders for white Guinea yam (*Dioscorea rotundata*) frequently suffer from open-loop, fixed-timer over-shearing, leading to starch matrix degradation, dough softening, and excessive energy consumption. This repository provides the computational framework to predict mechanical resistance (TPA hardness and adhesiveness) from multi-variable physicochemical and operational literature-derived features using advanced bagging and boosting regressors alongside Support Vector Regression (SVR).

## 📂 Repository Contents
- `rotary_yam_pounder_dataset.csv` (or Table S1): The structured meta-dataset comprising N = 120 observations extracted from peer-reviewed literature across diverse thermal and mechanical processing conditions.
- `rotary_pounder_pipeline.py` (or Jupyter Notebook): The complete Python implementation covering:
  - k-Nearest Neighbours (KNN, k=5) imputation for missing biochemical values.
  - One-Hot Encoding for processing modalities (mortar, rotary beater, attrition mill).
  - StandardScaler feature normalization fitted strictly on training subsets.
  - 10-fold cross-validation and hyperparameter tuning via GridSearchCV for Random Forest (RF), Gradient Boosting Regressor (GBR), and Support Vector Regressor (SVR).
  - SHapley Additive exPlanations (SHAP) feature attribution and dependence plotting.
- **Supplementary Tables (Tables S1–S5):** Detailed imputation diagnostics, hyperparameters, and full performance metrics.

## 🛠️ Tech Stack & Dependencies
- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`

## 🚀 Usage & Reproduction
1. Clone the repository:
   ```bash
   git clone (https://github.com/samdad0/rotary-pounder-regression)
