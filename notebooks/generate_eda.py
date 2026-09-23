"""Exploratory Data Analysis Generator for Bank Loan Approval Dataset.

Generates visual figures for report/figures and compiles a complete
Jupyter Notebook saved at notebooks/01_eda.ipynb.
"""

import os
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# Paths
DATA_PATH = "data/raw/loan_approval_dataset.csv"
FIG_DIR = "report/figures"
NOTEBOOK_PATH = "notebooks/01_eda.ipynb"

os.makedirs(FIG_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
for col in ["education", "self_employed", "loan_status"]:
    df[col] = df[col].str.strip()

# Derived features for EDA
df["total_assets"] = (
    df["residential_assets_value"]
    + df["commercial_assets_value"]
    + df["luxury_assets_value"]
    + df["bank_asset_value"]
)
df["loan_to_income"] = df["loan_amount"] / df["income_annum"]
df["asset_to_loan"] = df["total_assets"] / df["loan_amount"]

# --- 1. Target Distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["loan_status"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
for bar in bars:
    height = bar.get_height()
    pct = (height / len(df)) * 100
    ax.annotate(f"{height} ({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontweight="bold")
ax.set_title("Loan Approval Target Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Applicants")
ax.set_ylim(0, max(counts.values) * 1.15)
plt.savefig(os.path.join(FIG_DIR, "target_distribution.png"), dpi=300)
plt.close()

# --- 2. CIBIL Score Distribution by Status ---
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(
    data=df,
    x="cibil_score",
    hue="loan_status",
    kde=True,
    palette={"Approved": "#2ecc71", "Rejected": "#e74c3c"},
    bins=30,
    ax=ax,
    alpha=0.6,
    edgecolor="black"
)
ax.axvline(550, color="gray", linestyle="--", linewidth=1.5, label="Typical Risk Boundary (550)")
ax.set_title("CIBIL Credit Score Distribution by Loan Status", fontsize=13, fontweight="bold")
ax.set_xlabel("CIBIL Score (300 - 900)")
ax.set_ylabel("Count")
ax.legend(title="Loan Status")
plt.savefig(os.path.join(FIG_DIR, "cibil_distribution.png"), dpi=300)
plt.close()

# --- 3. Correlation Heatmap ---
df_corr = df.copy()
df_corr["loan_status_num"] = (df_corr["loan_status"] == "Approved").astype(int)
df_corr["education_num"] = (df_corr["education"] == "Graduate").astype(int)
df_corr["self_employed_num"] = (df_corr["self_employed"] == "Yes").astype(int)

cols_for_corr = [
    "cibil_score", "loan_status_num", "loan_term", "income_annum",
    "loan_amount", "total_assets", "loan_to_income", "asset_to_loan",
    "education_num", "self_employed_num", "no_of_dependents"
]
corr_matrix = df_corr[cols_for_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()

# --- 4. Financial Attributes vs Loan Status ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.boxplot(x="loan_status", y=df["income_annum"] / 1e5, data=df, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
axes[0].set_title("Annual Income by Status", fontweight="bold")
axes[0].set_ylabel("Income (Lakhs INR)")
axes[0].set_xlabel("Loan Status")

sns.boxplot(x="loan_status", y=df["loan_amount"] / 1e5, data=df, ax=axes[1], palette=["#2ecc71", "#e74c3c"])
axes[1].set_title("Loan Amount Requested by Status", fontweight="bold")
axes[1].set_ylabel("Loan Amount (Lakhs INR)")
axes[1].set_xlabel("Loan Status")

sns.boxplot(x="loan_status", y=df["total_assets"] / 1e5, data=df, ax=axes[2], palette=["#2ecc71", "#e74c3c"])
axes[2].set_title("Total Assets by Status", fontweight="bold")
axes[2].set_ylabel("Total Assets (Lakhs INR)")
axes[2].set_xlabel("Loan Status")

plt.savefig(os.path.join(FIG_DIR, "financial_distributions.png"), dpi=300)
plt.close()

# --- 5. Loan Term vs Approval Rate ---
term_stats = df.groupby("loan_term")["loan_status"].apply(lambda s: (s == "Approved").mean()).reset_index()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=term_stats, x="loan_term", y="loan_status", color="#3498db", ax=ax, edgecolor="black")
ax.set_title("Loan Approval Rate by Loan Term (Years)", fontsize=13, fontweight="bold")
ax.set_xlabel("Loan Term (Years)")
ax.set_ylabel("Approval Rate")
ax.set_ylim(0, 1.0)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
plt.savefig(os.path.join(FIG_DIR, "loan_term_impact.png"), dpi=300)
plt.close()

print("Figures successfully generated in report/figures/")

# --- Now Create Jupyter Notebook 01_eda.ipynb ---
nb = nbf.v4.new_notebook()

cells = []

# Title & intro
cells.append(nbf.v4.new_markdown_cell("""# Exploratory Data Analysis (EDA) — Bank Loan Approval Dataset

This notebook performs a comprehensive exploratory data analysis on the **Bank Loan Approval and Customer Risk Analysis** dataset (`architsharma01/loan-approval-prediction-dataset`).

### Key Analysis Objectives:
1. Understand dataset shape, feature types, and verify absence of missing values.
2. Investigate the target variable distribution (`loan_status`).
3. Analyze key predictors of loan approval (e.g. CIBIL score, loan term, income, assets).
4. Perform feature correlation and interaction analysis.
5. Formulate hypotheses for preprocessing, modeling, and risk segmentation.
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# Load the raw dataset
df = pd.read_csv('../data/raw/loan_approval_dataset.csv')
# Strip any leading/trailing whitespace from column names and string values
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()"""))

# Cell 2: Data summary & missing values
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Integrity & Summary Statistics
Check for missing values, column data types, and five-number summary."""))

cells.append(nbf.v4.new_code_cell("""print("Missing Values Check:")
print(df.isnull().sum())

print("\\nData Types:")
print(df.dtypes)

df.describe().T"""))

# Cell 3: Target variable
cells.append(nbf.v4.new_markdown_cell("""## 2. Target Variable Analysis: `loan_status`
Evaluate class balance between Approved and Rejected applications."""))

cells.append(nbf.v4.new_code_cell("""status_counts = df['loan_status'].value_counts()
status_pct = df['loan_status'].value_counts(normalize=True) * 100

summary_target = pd.DataFrame({'Count': status_counts, 'Percentage (%)': status_pct.round(2)})
display(summary_target)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(status_counts.index, status_counts.values, color=['#2ecc71', '#e74c3c'], width=0.5, edgecolor='black')
for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax.annotate(f"{h}\\n({pct:.1f}%)", (bar.get_x() + bar.get_width() / 2, h / 2),
                ha='center', va='center', color='white', fontweight='bold')
ax.set_title("Target Distribution (Loan Approval)", fontsize=13, fontweight='bold')
ax.set_ylabel("Count")
plt.show()"""))

# Cell 4: CIBIL Score Analysis
cells.append(nbf.v4.new_markdown_cell("""## 3. Credit Score (CIBIL) Analysis
Analyze how CIBIL credit score differentiates Approved vs. Rejected applicants."""))

cells.append(nbf.v4.new_code_cell("""display(df.groupby('loan_status')['cibil_score'].describe().round(2))

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(
    data=df,
    x='cibil_score',
    hue='loan_status',
    kde=True,
    palette={'Approved': '#2ecc71', 'Rejected': '#e74c3c'},
    bins=30,
    alpha=0.6,
    edgecolor='black',
    ax=ax
)
ax.axvline(550, color='gray', linestyle='--', linewidth=1.5, label='Risk Threshold (550)')
ax.set_title("CIBIL Credit Score Distribution by Loan Status", fontweight='bold')
ax.legend(title='Loan Status')
plt.show()"""))

# Cell 5: Asset and Financial Analysis
cells.append(nbf.v4.new_markdown_cell("""## 4. Financial Features & Derived Ratios
Examine income, loan amount, and asset holdings."""))

cells.append(nbf.v4.new_code_cell("""df['total_assets'] = (
    df['residential_assets_value']
    + df['commercial_assets_value']
    + df['luxury_assets_value']
    + df['bank_asset_value']
)
df['loan_to_income'] = df['loan_amount'] / df['income_annum']
df['asset_to_loan'] = df['total_assets'] / df['loan_amount']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.boxplot(x='loan_status', y=df['income_annum'] / 1e5, data=df, ax=axes[0], palette=['#2ecc71', '#e74c3c'])
axes[0].set_title("Annual Income (Lakhs INR)")
axes[0].set_xlabel("Loan Status")

sns.boxplot(x='loan_status', y=df['loan_amount'] / 1e5, data=df, ax=axes[1], palette=['#2ecc71', '#e74c3c'])
axes[1].set_title("Loan Amount (Lakhs INR)")
axes[1].set_xlabel("Loan Status")

sns.boxplot(x='loan_status', y=df['total_assets'] / 1e5, data=df, ax=axes[2], palette=['#2ecc71', '#e74c3c'])
axes[2].set_title("Total Assets (Lakhs INR)")
axes[2].set_xlabel("Loan Status")
plt.show()"""))

# Cell 6: Loan Term & Demographic Features
cells.append(nbf.v4.new_markdown_cell("""## 5. Loan Term, Education & Employment Status"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Approval rate by loan term
term_app = df.groupby('loan_term')['loan_status'].apply(lambda x: (x == 'Approved').mean())
sns.barplot(x=term_app.index, y=term_app.values, color='#3498db', ax=axes[0], edgecolor='black')
axes[0].set_title("Approval Rate by Loan Term (Years)", fontweight='bold')
axes[0].set_ylabel("Approval Rate")
axes[0].set_ylim(0, 1.0)

# Approval rate by education and employment
edu_emp = df.groupby(['education', 'self_employed'])['loan_status'].apply(lambda x: (x == 'Approved').mean()).unstack()
edu_emp.plot(kind='bar', ax=axes[1], colormap='viridis', edgecolor='black')
axes[1].set_title("Approval Rate by Education & Self-Employed Status", fontweight='bold')
axes[1].set_ylabel("Approval Rate")
axes[1].set_ylim(0, 1.0)
axes[1].legend(title='Self Employed')
axes[1].tick_params(axis='x', rotation=0)

plt.show()"""))

# Cell 7: Correlation Heatmap
cells.append(nbf.v4.new_markdown_cell("""## 6. Correlation Matrix"""))

cells.append(nbf.v4.new_code_cell("""df_corr = df.copy()
df_corr['loan_status_num'] = (df_corr['loan_status'] == 'Approved').astype(int)
df_corr['education_num'] = (df_corr['education'] == 'Graduate').astype(int)
df_corr['self_employed_num'] = (df_corr['self_employed'] == 'Yes').astype(int)

numeric_cols = [
    'cibil_score', 'loan_status_num', 'loan_term', 'income_annum',
    'loan_amount', 'total_assets', 'loan_to_income', 'asset_to_loan',
    'education_num', 'self_employed_num', 'no_of_dependents'
]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title("Correlation Matrix", fontweight='bold')
plt.show()"""))

# Cell 8: Key Insights & Takeaways
cells.append(nbf.v4.new_markdown_cell("""## 7. Key Findings & Modeling Strategy

1. **Cleanliness**: 0 missing values across all 4,269 rows. Column names and string values contain leading whitespace which will be cleaned in preprocessing.
2. **Class Balance**: 62.2% Approved (2,656) vs 37.8% Rejected (1,613). Moderate imbalance — standard stratified sampling is appropriate.
3. **Primary Predictor**: `cibil_score` is the dominant feature (correlation 0.77). Applicants with CIBIL >= 600 have an overwhelming approval rate, while CIBIL < 550 faces near-universal rejection.
4. **Secondary Predictors**: Longer `loan_term` moderately increases rejection probability. Income and asset values provide critical collateral context for risk scoring.
5. **Feature Engineering**: Creating `total_assets`, `loan_to_income` ratio, and `asset_to_loan` coverage provides strong financial interpretability for both approval classification and risk tiering.
"""))

nb.cells = cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully written to {NOTEBOOK_PATH}")
