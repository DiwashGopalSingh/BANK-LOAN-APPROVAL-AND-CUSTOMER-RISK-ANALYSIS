"""Loan Approval Classification Training and Model Comparison.

Trains, evaluates, and compares multiple classification algorithms:
1. Logistic Regression (with StandardScaler)
2. Decision Tree Classifier
3. Random Forest Classifier

Saves evaluation metrics, diagnostic plots, and serializes the champion model to models/.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import (
    compute_classification_metrics,
    format_metrics_table,
    plot_confusion_matrices,
    plot_roc_curves,
)
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

# ==========================================
# Named Constants
# ==========================================
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
MODEL_SAVE_PATH = "models/loan_approval_model.pkl"
COMPARISON_CSV_PATH = "models/model_comparison.csv"
FIGURES_DIR = "report/figures"

RANDOM_SEED = 42
MAX_ITER = 1000


def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads processed train and test datasets from data/processed/.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test.
    """
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def get_candidate_models() -> Dict[str, Any]:
    """Instantiates the candidate classification models for comparison.

    Returns:
        Dictionary of model names mapped to their respective estimators/pipelines.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=RANDOM_SEED, max_iter=MAX_ITER)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def train_and_evaluate_all() -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """Fits all candidate models, computes metrics, generates plots, and identifies champion.

    Returns:
        Tuple of (trained_models_dict, comparison_dataframe, champion_model_name).
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train, X_test, y_train, y_test = load_train_test_data()
    candidates = get_candidate_models()

    trained_models: Dict[str, Any] = {}
    metrics_results: List[Dict[str, Any]] = []
    roc_items: List[Dict[str, Any]] = []

    print("=" * 60)
    print("STARTING MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    for name, model in candidates.items():
        print(f"\nTraining [{name}] on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predictions on holdout test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_classification_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=name,
        )
        metrics_results.append(metrics)
        roc_items.append({"model_name": name, "y_prob": y_prob})

        print(f" -> {name} - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    # Generate comparison table
    comparison_df = format_metrics_table(metrics_results)
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON (TEST SET)")
    print("=" * 60)
    print(comparison_df.to_string())
    print("=" * 60)

    comparison_df.to_csv(COMPARISON_CSV_PATH)
    print(f"Metrics table saved to {COMPARISON_CSV_PATH}")

    # Visualizations
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plot_confusion_matrices(metrics_results, save_path=cm_path)
    print(f"Confusion matrices plot saved to {cm_path}")

    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plot_roc_curves(roc_items, y_test.values, save_path=roc_path)
    print(f"ROC curves plot saved to {roc_path}")

    # Feature Importance for Tree Models
    rf_model = trained_models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)

    plt.figure(figsize=(9, 6))
    importances.plot(kind="barh", color="#2980b9", edgecolor="black")
    plt.title("Random Forest Feature Importances", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {fi_path}")

    # Champion selection: Random Forest is preferred for production underwriting due to
    # ensemble bagging robustness against single-tree overfitting and continuous probability calibration.
    champion_name = "Random Forest"
    print(f"\nChampion Model Selected: [{champion_name}] (Ensemble Robustness, F1-Score: {comparison_df.loc[champion_name, 'F1-Score']})")

    # Serialize model artifact
    champion_model = trained_models[champion_name]
    artifact = {
        "model_name": champion_name,
        "model": champion_model,
        "all_models": trained_models,
        "feature_names": FEATURE_COLUMNS,
        "comparison_metrics": comparison_df.to_dict(orient="index"),
        "champion_metrics": comparison_df.loc[champion_name].to_dict(),
        "feature_importances": importances.to_dict(),
    }

    joblib.dump(artifact, MODEL_SAVE_PATH)
    print(f"Model artifact serialized to {MODEL_SAVE_PATH}")

    return trained_models, comparison_df, champion_name


if __name__ == "__main__":
    train_and_evaluate_all()
