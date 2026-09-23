# 🏦 Bank Loan Approval & Customer Risk Analysis

An end-to-end Data Mining and Applied Machine Learning system that predicts loan approval and evaluates applicant risk profiles using multidimensional credit underwriting metrics. Built with **scikit-learn**, **pandas**, and **Streamlit**.

---

## 🌟 Key Features

1. **Loan Approval Classification**:
   - Compares **Logistic Regression**, **Decision Tree**, and **Random Forest** algorithms.
   - Evaluated on 4,269 historical records with stratified 80/20 train/test splits.
   - Operational champion: **Random Forest** (99.77% accuracy, 0.9981 F1-score, 1.0000 ROC-AUC) providing continuous, calibrated confidence scores.

2. **Multidimensional Customer Risk Scoring**:
   - Independent underwriting evaluation combining:
     - **Creditworthiness**: CIBIL bureau score rating.
     - **Debt Burden**: Loan-to-income (LTI) ratio.
     - **Collateral Protection**: Total assets-to-loan coverage ratio.
   - Categorizes applicants into **Low Risk**, **Medium Risk**, and **High Risk** tiers with natural-language driving factors.
   - Includes unsupervised **K-Means Clustering** ($k=3$) customer persona segmentation.

3. **Interactive Streamlit Web Dashboard**:
   - Clean, single-session interface to input applicant details.
   - Instant dual output: Approval status + confidence percentage, alongside risk tier + risk factors.
   - Interactive model diagnostics (confusion matrices, ROC curves, and feature importance).

---

## 📁 Repository Structure

```
.
├── Context/                  # Project specifications and architecture rules
├── data/
│   ├── raw/                  # Original untouched dataset (Kaggle)
│   └── processed/            # Cleaned data and stratified train/test splits
├── notebooks/                # Exploratory Data Analysis (01_eda.ipynb)
├── src/
│   ├── preprocessing.py      # Cleaning, encoding, and financial feature engineering
│   ├── train_approval.py     # Multi-model training and evaluation pipeline
│   ├── train_risk.py         # Rule-based scoring and K-Means customer segmentation
│   └── evaluate.py           # Metric calculations and plotting utilities
├── models/                   # Serialized model artifacts (.pkl)
├── report/                   # Final written academic/technical report & figures
├── app.py                    # Streamlit web application
├── requirements.txt          # Pinned project dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Performance Summary

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 91.33% | 0.9193 | 0.9435 | 0.9312 | 0.9729 |
| **Decision Tree** | 99.88% | 0.9981 | 1.0000 | 0.9991 | 1.0000 |
| **Random Forest (Champion)** | **99.77%** | **0.9962** | **1.0000** | **0.9981** | **1.0000** |

---

## 📑 Full Report & Findings

For complete methodological details, exploratory analysis visualizations, and credit risk calibrations, see the formal project report:
👉 [`report/loan_approval_risk_report.md`](report/loan_approval_risk_report.md)
