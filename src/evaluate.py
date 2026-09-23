"""Model Evaluation and Performance Metrics Module.

Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
plots confusion matrices and ROC curves, and generates comparison tables.
"""

import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

FIGURES_DIR = "report/figures"


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics for binary classification.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probability of positive class (for ROC-AUC).
        model_name: Name of the algorithm being evaluated.

    Returns:
        Dictionary of computed metric values.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

    cm = confusion_matrix(y_true, y_pred)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def format_metrics_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Formats a list of metric dictionaries into a scannable comparison DataFrame.

    Args:
        results_list: List of metric dicts returned by compute_classification_metrics.

    Returns:
        Pandas DataFrame indexed by model name with rounded metrics.
    """
    rows = []
    for r in results_list:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1_score"], 4),
            "ROC-AUC": round(r["roc_auc"], 4),
        })

    return pd.DataFrame(rows).set_index("Model")


def plot_confusion_matrices(results_list: List[Dict[str, Any]], save_path: str = None) -> None:
    """Plots side-by-side confusion matrix heatmaps for all evaluated models.

    Args:
        results_list: List of metric dicts containing 'model_name' and 'confusion_matrix'.
        save_path: Filepath where the plot will be saved.
    """
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    labels = ["Rejected (0)", "Approved (1)"]

    for i, res in enumerate(results_list):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[i],
            cbar=False,
        )
        axes[i].set_title(f"{res['model_name']}\nAccuracy: {res['accuracy']:.2%}", fontweight="bold")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_roc_curves(models_probs: List[Dict[str, Any]], y_true: np.ndarray, save_path: str = None) -> None:
    """Plots superimposed ROC curves with AUC values for model comparison.

    Args:
        models_probs: List of dicts with 'model_name' and 'y_prob'.
        y_true: True binary target array.
        save_path: Filepath to save ROC curves plot.
    """
    plt.figure(figsize=(7, 6))

    for item in models_probs:
        fpr, tpr, _ = roc_curve(y_true, item["y_prob"])
        auc = roc_auc_score(y_true, item["y_prob"])
        plt.plot(fpr, tpr, label=f"{item['model_name']} (AUC = {auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
