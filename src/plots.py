"""
plots.py
Generates data visualizations and handles testing on unseen samples.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_and_save_correlation_heatmap(
    df: pd.DataFrame, output_path: str = "outputs/correlation_heatmap.png"
):
    """Generates and saves a correlation heatmap for numerical features in the dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def plot_and_save_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "outputs/confusion_matrix.png",
):
    """Generates and saves the Confusion Matrix plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def plot_and_save_roc_curve(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = "outputs/roc_curve.png",
):
    """Generates and saves the ROC Curve plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.title("ROC Curve")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def test_unseen_sample(model, unseen_df: pd.DataFrame, expected_columns: list):
    """Preprocesses a raw unseen data sample and outputs prediction results."""
    # 1. Basic Cleaning
    df_clean = unseen_df.copy()
    if "company" in df_clean.columns:
        df_clean = df_clean.drop(columns=["company"])
    df_clean["agent"] = df_clean.get("agent", pd.Series()).fillna(0)
    df_clean["country"] = df_clean.get("country", pd.Series()).fillna("Unknown")
    df_clean["children"] = df_clean.get("children", pd.Series()).fillna(0)

    # 2. One-hot encode
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    # 3. Align columns to match training set features
    df_aligned = df_encoded.reindex(columns=expected_columns, fill_value=0)

    # 4. Predict class and cancellation probability
    prediction = model.predict(df_aligned)[0]
    probability = model.predict_proba(df_aligned)[0, 1]
