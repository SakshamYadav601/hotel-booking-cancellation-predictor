"""
plots.py
Responsible for generating and saving Confusion Matrix and ROC Curve plots.
Lead: Member 4 / Visualizations
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_and_save_visualizations(model, X_eval: pd.DataFrame, y_eval: pd.Series, prefix: str = "test", output_dir: str = "reports/figures"):
    """
    Generates and exports Confusion Matrix and ROC Curve plots for a dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_eval, y_eval, cmap="Blues")
    plt.title(f"Confusion Matrix ({prefix.capitalize()} Data)")
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Saved Confusion Matrix plot to {cm_path}")
    
    # 2. ROC Curve Plot
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_eval, y_eval)
    plt.title(f"ROC Curve ({prefix.capitalize()} Data)")
    roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Saved ROC Curve plot to {roc_path}")
    