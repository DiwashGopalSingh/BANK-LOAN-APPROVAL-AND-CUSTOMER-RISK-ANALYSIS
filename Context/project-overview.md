# Project Overview — Bank Loan Approval and Customer Risk Analysis

## Domain
Data Mining

## Objective
Build a system that:
1. Classifies whether a loan application should be **approved or rejected**.
2. Assesses **customer risk** (a risk tier — low / medium / high) based on applicant financial data.

Both are surfaced through a lightweight web app, backed by a written report documenting the process and findings.

## Deliverables
- Trained classification model(s) for loan approval
- A risk-scoring component (rule-based tiers or clustering-based segmentation)
- A Streamlit app: input applicant details → get approval decision + risk tier
- Written report: problem statement, methodology, EDA findings, model comparison, results, conclusion

## In Scope
- Data cleaning & EDA
- Loan approval classification — compare 2–3 algorithms (Logistic Regression, Decision Tree, Random Forest)
- Risk analysis — derived from credit score / asset features, or via k-means clustering for customer segmentation
- Model evaluation (accuracy, precision, recall, F1, confusion matrix) with a written comparison
- A simple, functional UI (Streamlit)
- The written report

## Out of Scope
- Any real bank integration or live data feeds
- Authentication, multi-user roles, production-grade security
- Public/cloud deployment — local or demo hosting is enough
- Real customer PII — public/synthetic data only

## Dataset
Kaggle: `architsharma01/loan-approval-prediction-dataset` (~4,269 rows) — chosen because it includes both `loan_status` (for classification) and `cibil_score` / asset value fields (for risk analysis), so one dataset covers both halves of the project.

Fallback: the classic `altruistdelhite04/loan-prediction-problem-dataset` (614 rows) if a lighter dataset is preferred.

## Tech Stack
- Python, pandas, scikit-learn — data mining core
- Streamlit — UI
- matplotlib / seaborn — EDA visuals

## Success Criteria
- Both models (approval + risk) run end-to-end on the chosen dataset
- App takes applicant input and returns both outputs without errors
- Report clearly explains methodology and justifies model choices
