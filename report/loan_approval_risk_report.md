# Bank Loan Approval and Customer Risk Analysis
**A Data Mining & Applied Machine Learning Study**

---

## Executive Summary

Credit risk assessment and automated loan underwriting are pivotal pillars of modern retail banking. Manual loan underwriting is time-consuming, prone to human inconsistency, and often fails to capture multi-attribute financial risk profiles. This project delivers an end-to-end data mining system designed to:
1. **Automate Loan Approval Classification**: Accurately predict whether a retail loan application should be approved or rejected based on applicant financial, demographic, and credit attributes.
2. **Quantify Multidimensional Customer Risk**: Assess applicant credit risk into transparent, actionable risk tiers (**Low Risk**, **Medium Risk**, **High Risk**) with natural-language explainability and empirical unsupervised customer segmentation.
3. **Operationalize via an Interactive Web Interface**: Provide loan officers and applicants with an intuitive, real-time decision dashboard powered by **Streamlit**.

The system is trained and evaluated on 4,269 historical loan application records. Across rigorous holdout evaluation, machine learning models demonstrated exceptional predictive capability, with the champion tree-based classifier achieving **99.88% accuracy** and a **0.9991 F1-score**, driven predominantly by non-linear credit threshold interactions.

---

## 1. Problem Formulation & Objectives

### 1.1 Business Context
Financial institutions face the dual challenge of maximizing loan portfolio profitability while minimizing default and non-performing asset (NPA) rates. Granting loans to subprime applicants results in severe capital losses, while erroneously rejecting creditworthy applicants incurs substantial opportunity costs.

### 1.2 Core Objectives
- **Data Mining & EDA**: Identify the primary drivers of loan approval and discover structural correlations between income, debt amount, asset portfolios, and approval status.
- **Comparative Supervised Modeling**: Formulate a supervised binary classification problem ($y \in \{0, 1\}$) and benchmark multiple classification algorithms:
  - Logistic Regression (Linear baseline with feature scaling)
  - Decision Tree Classifier (Non-linear rule-based learner)
  - Random Forest Classifier (Ensemble bagging estimator)
- **Explainable Risk Profiling**: Design an independent risk assessment framework evaluating credit score, debt-to-income (LTI) ratio, and asset-to-loan collateral coverage.
- **Decision Support Deployment**: Deliver a zero-latency Streamlit application allowing single-point underwriting evaluation with real-time risk factor explanations.

---

## 2. Dataset Characterization & Exploratory Data Analysis

### 2.1 Dataset Overview
The project utilizes the Kaggle `architsharma01/loan-approval-prediction-dataset`.

| Attribute | Data Type | Description |
|---|---|---|
| `loan_id` | Identifier | Unique applicant ID (dropped during modeling) |
| `no_of_dependents` | Integer (0 – 5) | Number of family dependents |
| `education` | Categorical | Educational attainment (`Graduate`, `Not Graduate`) |
| `self_employed` | Categorical | Employment type (`Yes`, `No`) |
| `income_annum` | Integer | Annual gross income (₹1,00,000 – ₹99,00,000) |
| `loan_amount` | Integer | Total loan principal requested (₹3,00,000 – ₹3,95,00,000) |
| `loan_term` | Integer (2 – 20) | Loan repayment period in years |
| `cibil_score` | Integer (300 – 900) | Credit Bureau score |
| `residential_assets_value` | Integer | Assessed value of residential properties |
| `commercial_assets_value` | Integer | Assessed value of commercial properties |
| `luxury_assets_value` | Integer | Value of luxury assets (vehicles, jewelry, etc.) |
| `bank_asset_value` | Integer | Liquid assets held in bank deposits |
| `loan_status` | Categorical (Target) | Underwriting outcome (`Approved`, `Rejected`) |

### 2.2 Data Quality & Hygiene
- **Missing Values**: 0 null or missing values across all 4,269 rows and 13 columns.
- **String Formatting**: Raw CSV values contained leading and trailing whitespace characters (e.g. `' Approved'`, `' Graduate'`), which were systematically stripped during preprocessing.
- **Class Balance**:
  - **Approved**: 2,656 instances (62.22%)
  - **Rejected**: 1,613 instances (37.78%)
  - The dataset exhibits moderate class balance, requiring stratified train-test splitting to maintain representative target distributions.

### 2.3 Exploratory Insights
1. **The CIBIL Credit Score Divide**:
   - Approved loans show an average CIBIL score of **703.5** (median 711, IQR 618 – 803).
   - Rejected loans show an average CIBIL score of **429.5** (median 429, IQR 364 – 493).
   - The Pearson correlation between `cibil_score` and `loan_status` is **+0.771**, establishing credit bureau rating as the single most decisive factor.
2. **Loan Term Sensitivity**:
   - Longer loan tenures correlate negatively with approval ($\rho = -0.113$). Applicants requesting terms exceeding 14 years experience elevated rejection rates due to long-term default exposure.
3. **Asset Portfolios & Income Distribution**:
   - Income and asset values exhibit near-identical distributions between approved and rejected classes, indicating that in traditional banking heuristic data, high assets alone do not compensate for a deteriorated credit score.

---

## 3. Data Preprocessing & Feature Engineering

To provide richer financial context for underwriting algorithms, three domain-specific composite features were engineered:

$$\text{Total Assets} = \text{Residential} + \text{Commercial} + \text{Luxury} + \text{Bank Assets}$$

$$\text{Loan-to-Income (LTI)} = \frac{\text{Loan Amount}}{\text{Annual Income}}$$

$$\text{Asset Coverage Ratio} = \frac{\text{Total Assets}}{\text{Loan Amount}}$$

### 3.1 Pipeline Design & Data Leakage Prevention
- **Train/Test Splitting**: Stratified 80/20 train/test split (3,415 training samples, 854 test samples) using fixed random seed (`random_state=42`).
- **Feature Scaling**: Numerical attributes for linear models are scaled using `StandardScaler` inside a scikit-learn `Pipeline` fitted strictly on training data.
- **Shared Inference Bridge**: Feature preparation logic is encapsulated in `src.preprocessing.prepare_applicant_features()`, guaranteeing that web app user submissions undergo identical transformations without code duplication.

---

## 4. Loan Approval Model Benchmark & Evaluation

### 4.1 Candidate Algorithms
Three diverse classification families were trained and tested on the holdout evaluation set (854 records):
1. **Logistic Regression (Baseline Linear Classifier)**:
   - Optimized with L2 regularization and standard feature normalization.
2. **Decision Tree Classifier (Non-linear Rule Induction)**:
   - Configured with `max_depth=6`, `min_samples_split=10`, `min_samples_leaf=5` to prevent memorization and leaf overfitting.
3. **Random Forest Classifier (Ensemble Bagging)**:
   - Ensemble of 150 randomized decision trees with bootstrap aggregation and sub-feature sampling (`max_depth=8`, `min_samples_split=6`).

### 4.2 Benchmark Results

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 91.33% | 0.9193 | 0.9435 | 0.9312 | 0.9729 |
| **Decision Tree** | **99.88%** | **0.9981** | **1.0000** | **0.9991** | **1.0000** |
| **Random Forest** | 99.77% | 0.9962 | 1.0000 | 0.9981 | 1.0000 |

### 4.3 Analysis of Model Performance
- **Linear vs Non-Linear Performance**: While Logistic Regression performs commendably (91.33% accuracy), it struggles to model the sharp step-function threshold of the credit score boundary.
- **Tree Dominance**: Both the Decision Tree and Random Forest achieved near-perfect separation. The Decision Tree achieved a test accuracy of **99.88%** with only a single false positive on the holdout test set (531 true approvals correctly identified with 100% recall).
- **Champion Selection**: The **Random Forest** classifier was serialized as the primary operational model due to its ensemble bagging robustness against single-tree overfitting and continuous probability calibration (99.77% accuracy, 0.9981 F1, 1.0000 ROC-AUC). Furthermore, the system serializes all three trained estimators into the artifact, allowing loan officers to interactively toggle between Random Forest, Decision Tree, and Logistic Regression directly within the Streamlit UI.

---

## 5. Multidimensional Customer Risk Scoring & Segmentation

In accordance with banking best practices and project architectural separation, customer risk scoring operates as an independent component decoupled from the binary approval model.

### 5.1 The Credit Underwriting Triad
Applicant financial health is evaluated across three core risk dimensions (Total Score: 0 – 100, where lower indicates lower default risk):

1. **Credit History (CIBIL Score — Max 50 pts)**:
   - $\ge 750$: 0 pts (Prime borrower)
   - $650 - 749$: 15 pts (Good credit standing)
   - $550 - 649$: 30 pts (Moderate credit profile)
   - $< 550$: 50 pts (Subprime borrower — severe default risk)
2. **Debt Service Capacity (Loan-to-Income — Max 25 pts)**:
   - $\le 2.0\text{x}$: 0 pts (Conservative leverage)
   - $2.0 - 3.5\text{x}$: 12 pts (Moderate debt burden)
   - $> 3.5\text{x}$: 25 pts (High debt burden)
3. **Collateral Protection (Asset Coverage Ratio — Max 25 pts)**:
   - $\ge 1.5\text{x}$: 0 pts (Strong collateral coverage)
   - $1.0 - 1.5\text{x}$: 10 pts (Adequate coverage)
   - $< 1.0\text{x}$: 25 pts (Under-collateralized loan)

### 5.2 Calibrated Risk Tiers
- **Low Risk ($\le 30$ pts)**: 35.51% of applicants — Prime credit, solid asset backing, and conservative debt ratios.
- **Medium Risk ($31 - 60$ pts)**: 22.49% of applicants — Moderate debt leverage or borderline credit score requiring loan covenant adjustments.
- **High Risk ($> 60$ pts)**: 42.00% of applicants — Severely impaired credit scores and/or under-collateralized loans.

### 5.3 Unsupervised K-Means Customer Personas
Applying unsupervised K-Means clustering ($k=3$) over normalized financial attributes corroborated the empirical distribution:
- **Cluster 0 — Subprime Tier (37.55%)**: Characterized by low mean CIBIL scores (430) and high rejection rates.
- **Cluster 1 — Moderate Tier (35.49%)**: Characterized by mid-range CIBIL scores (640) and balanced asset portfolios.
- **Cluster 2 — Prime Tier (26.96%)**: Characterized by top-tier CIBIL scores (780+) and high net-worth asset cushions.

---

## 6. Streamlit Web Application

The system was operationalized in `app.py`, offering a single-session decisioning interface:
- **Input Form**: User-friendly input widgets for personal, loan, credit, and asset details with input constraints matching domain ranges.
- **Underwriting Decision Card**: Displays immediate **Approved / Rejected** determination alongside model confidence percentage.
- **Risk Profile Card**: Displays color-coded risk tier badges (**🟢 Low Risk**, **🟡 Medium Risk**, **🔴 High Risk**) and itemizes driving risk factors (e.g. *"Subprime credit score (CIBIL: 420)"*, *"High debt burden (4.2x annual income)"*).
- **Embedded Analytics**: Interactive tabs providing model benchmark tables, feature importances, and diagnostic ROC / confusion matrix plots.

---

## 7. Limitations & Recommendations

1. **Synthetic Feature Thresholds**: In this benchmark dataset, CIBIL score dominates the classification label to an extreme degree. Real-world retail banking involves nuanced interactions such as debt-service-coverage ratios (DSCR), employment longevity, and verifiable tax return history.
2. **Dynamic Risk Weighting**: Future iterations can calibrate risk point assignments via logistic weight optimization or survival analysis for time-to-default modeling.
3. **Model Monitoring**: For real-world deployment, continuous monitoring for concept drift and population stability index (PSI) is recommended.

---

## 8. Conclusion

This project successfully developed, validated, and deployed a complete Data Mining solution for retail bank loan approval and customer risk analysis. By separating the supervised classification decision from the multidimensional risk scoring engine, the system delivers high accuracy (99.88%) alongside transparent, explainable decision support. All code, artifacts, notebooks, and models are fully reproducible within the project repository.
