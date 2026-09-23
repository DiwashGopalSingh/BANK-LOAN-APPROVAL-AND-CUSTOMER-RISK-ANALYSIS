"""Customer Risk Scoring and Segmentation Module.

Implements two complementary risk assessment approaches:
1. A Rule-Based Scoring Engine providing transparent Low/Medium/High risk tiers
   and human-interpretable risk driver explanations.
2. An Unsupervised K-Means Clustering model (k=3) for empirical customer segmentation.

Follows PEP 8 standards with named constants for all thresholds.
"""

import os
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing import FEATURE_COLUMNS, PROCESSED_DATA_PATH

# ==========================================
# Named Risk Threshold Constants
# ==========================================
# CIBIL Score Thresholds
CIBIL_PRIME = 750
CIBIL_GOOD = 650
CIBIL_FAIR = 550

# Loan to Income (LTI) Thresholds
LTI_CONSERVATIVE = 2.0
LTI_ELEVATED = 3.5

# Asset to Loan Coverage Thresholds
ASSET_COVERAGE_STRONG = 1.5
ASSET_COVERAGE_ADEQUATE = 1.0

# Composite Risk Score Boundaries (0 - 100)
RISK_SCORE_LOW_MAX = 30
RISK_SCORE_MED_MAX = 60

# Model Paths
RISK_MODEL_SAVE_PATH = "models/risk_scoring.pkl"
FIGURES_DIR = "report/figures"


def evaluate_risk_tier(
    cibil_score: float,
    loan_to_income: float,
    asset_to_loan: float,
) -> Dict[str, Any]:
    """Calculates composite risk score (0-100), risk tier, and driving risk factors.

    Args:
        cibil_score: Applicant credit score (300 - 900).
        loan_to_income: Ratio of loan amount to annual income.
        asset_to_loan: Ratio of total assets to loan amount.

    Returns:
        Dictionary with:
            - 'risk_score': Integer (0 to 100, higher means riskier).
            - 'risk_tier': 'Low Risk', 'Medium Risk', or 'High Risk'.
            - 'tier_color': Hex color code for UI display.
            - 'key_factors': List of driving explanations.
    """
    score = 0
    factors: List[str] = []

    # 1. CIBIL Score Component (Up to 50 points)
    if cibil_score >= CIBIL_PRIME:
        score += 0
        factors.append(f"Excellent credit history (CIBIL: {int(cibil_score)})")
    elif cibil_score >= CIBIL_GOOD:
        score += 15
        factors.append(f"Good credit score (CIBIL: {int(cibil_score)})")
    elif cibil_score >= CIBIL_FAIR:
        score += 30
        factors.append(f"Fair/Borderline credit score (CIBIL: {int(cibil_score)})")
    else:
        score += 50
        factors.append(f"Subprime credit score (CIBIL: {int(cibil_score)}) - elevated default risk")

    # 2. Loan to Income Ratio Component (Up to 25 points)
    if loan_to_income <= LTI_CONSERVATIVE:
        score += 0
        factors.append(f"Conservative loan-to-income ratio ({loan_to_income:.2f}x)")
    elif loan_to_income <= LTI_ELEVATED:
        score += 12
        factors.append(f"Moderate debt burden relative to income ({loan_to_income:.2f}x)")
    else:
        score += 25
        factors.append(f"High debt burden ({loan_to_income:.2f}x annual income)")

    # 3. Asset Collateral Coverage Component (Up to 25 points)
    if asset_to_loan >= ASSET_COVERAGE_STRONG:
        score += 0
        factors.append(f"Strong collateral backing ({asset_to_loan:.2f}x asset coverage)")
    elif asset_to_loan >= ASSET_COVERAGE_ADEQUATE:
        score += 10
        factors.append(f"Adequate asset coverage ({asset_to_loan:.2f}x)")
    else:
        score += 25
        factors.append(f"Under-collateralized loan ({asset_to_loan:.2f}x asset coverage)")

    # Final Risk Tier Determination
    score = min(score, 100)
    if score <= RISK_SCORE_LOW_MAX:
        tier = "Low Risk"
        color = "#27ae60"  # Green
    elif score <= RISK_SCORE_MED_MAX:
        tier = "Medium Risk"
        color = "#f39c12"  # Amber/Orange
    else:
        tier = "High Risk"
        color = "#c0392b"  # Red

    return {
        "risk_score": score,
        "risk_tier": tier,
        "tier_color": color,
        "key_factors": factors,
    }


def train_kmeans_segmentation(df: pd.DataFrame) -> Tuple[KMeans, StandardScaler, pd.DataFrame]:
    """Fits an unsupervised K-Means model (k=3) on applicant financial and credit features.

    Args:
        df: Processed DataFrame.

    Returns:
        Tuple of (fitted_kmeans_model, fitted_scaler, clustered_dataframe).
    """
    cluster_features = [
        "cibil_score",
        "loan_to_income",
        "asset_to_loan",
        "income_annum",
        "total_assets",
    ]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered["cluster"] = kmeans.fit_predict(scaled_data)

    # Characterize clusters by mean CIBIL score
    cluster_profiles = df_clustered.groupby("cluster")[cluster_features].mean()
    # Map cluster index to readable persona
    cibil_rank = cluster_profiles["cibil_score"].sort_values(ascending=False).index.tolist()
    cluster_names = {
        cibil_rank[0]: "Prime Tier (Low Risk)",
        cibil_rank[1]: "Moderate Tier (Medium Risk)",
        cibil_rank[2]: "Subprime Tier (High Risk)",
    }
    df_clustered["cluster_name"] = df_clustered["cluster"].map(cluster_names)

    return kmeans, scaler, df_clustered


def plot_risk_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """Plots risk tier distribution and clustering segmentation.

    Args:
        df: DataFrame with evaluated risk tiers.
        save_path: Filepath where the figure will be saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Rule-Based Risk Tier Breakdown
    tier_counts = df["risk_tier"].value_counts()
    tier_colors = {"Low Risk": "#27ae60", "Medium Risk": "#f39c12", "High Risk": "#c0392b"}
    colors = [tier_colors.get(t, "#95a5a6") for t in tier_counts.index]

    bars = axes[0].bar(tier_counts.index, tier_counts.values, color=colors, edgecolor="black", width=0.5)
    for b in bars:
        h = b.get_height()
        pct = (h / len(df)) * 100
        axes[0].annotate(f"{h}\n({pct:.1f}%)", (b.get_x() + b.get_width() / 2, h / 2),
                         ha="center", va="center", color="white", fontweight="bold")
    axes[0].set_title("Rule-Based Applicant Risk Tiers", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Applicant Count")

    # 2. CIBIL Score vs Asset-to-Loan by Risk Tier
    sns.scatterplot(
        data=df,
        x="cibil_score",
        y="asset_to_loan",
        hue="risk_tier",
        palette=tier_colors,
        alpha=0.6,
        ax=axes[1],
    )
    axes[1].set_title("Risk Tiers: CIBIL Score vs Collateral Ratio", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("CIBIL Credit Score")
    axes[1].set_ylabel("Asset-to-Loan Ratio")
    axes[1].set_ylim(0, 10)  # Zoom in for clear scatter visibility
    axes[1].axvline(550, color="gray", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def run_risk_pipeline() -> None:
    """Runs the risk scoring and segmentation pipeline, printing stats and saving artifacts."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("STARTING CUSTOMER RISK SCORING & SEGMENTATION PIPELINE")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 1. Apply Rule-Based Scoring across dataset
    risk_results = [
        evaluate_risk_tier(row["cibil_score"], row["loan_to_income"], row["asset_to_loan"])
        for _, row in df.iterrows()
    ]

    df["risk_score"] = [r["risk_score"] for r in risk_results]
    df["risk_tier"] = [r["risk_tier"] for r in risk_results]

    print("\nRule-Based Risk Tier Distribution:")
    tier_summary = df["risk_tier"].value_counts(normalize=True) * 100
    for tier, pct in tier_summary.items():
        count = (df["risk_tier"] == tier).sum()
        print(f" - {tier}: {count} applicants ({pct:.2f}%)")

    # 2. Run K-Means Clustering
    print("\nTraining K-Means (k=3) customer segmentation model...")
    kmeans, scaler, df_clustered = train_kmeans_segmentation(df)

    print("K-Means Cluster Breakdown:")
    for cluster_name, count in df_clustered["cluster_name"].value_counts().items():
        print(f" - {cluster_name}: {count} applicants ({count/len(df):.2%})")

    # 3. Save Visualizations
    fig_path = os.path.join(FIGURES_DIR, "customer_risk_analysis.png")
    plot_risk_distribution(df, save_path=fig_path)
    print(f"\nRisk distribution figure saved to {fig_path}")

    # 4. Serialize Risk Model & Scoring Engine
    risk_artifact = {
        "evaluate_risk_tier_fn": evaluate_risk_tier,
        "kmeans_model": kmeans,
        "scaler": scaler,
        "thresholds": {
            "cibil_prime": CIBIL_PRIME,
            "cibil_good": CIBIL_GOOD,
            "cibil_fair": CIBIL_FAIR,
            "lti_conservative": LTI_CONSERVATIVE,
            "lti_elevated": LTI_ELEVATED,
            "asset_coverage_strong": ASSET_COVERAGE_STRONG,
            "asset_coverage_adequate": ASSET_COVERAGE_ADEQUATE,
            "risk_score_low_max": RISK_SCORE_LOW_MAX,
            "risk_score_med_max": RISK_SCORE_MED_MAX,
        },
        "tier_summary": tier_summary.to_dict(),
    }

    joblib.dump(risk_artifact, RISK_MODEL_SAVE_PATH)
    print(f"Risk assessment artifact serialized to {RISK_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    run_risk_pipeline()
