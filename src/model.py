"""
model.py
Responsible for training the Logistic Regression model, 
making predictions, and returning evaluation metrics.
Lead: Model Development (Member 3)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Initializes and trains a Logistic Regression model on the training data.
    """
    print("[INFO] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("[INFO] Model training completed successfully.")
    return model


"""
evaluate.py
Responsible for evaluating the trained model on test and unseen holdout data,
generating metrics, and saving performance plots.
Lead: Model Evaluation (Member 4)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def evaluate_model(model, X_eval: pd.DataFrame, y_eval: pd.Series, dataset_name: str = "Test Data") -> dict:
    """
    Generates predictions on any evaluation dataset and computes core metrics.
    """
    print(f"\n[INFO] Evaluating model performance on [{dataset_name}]...")
    
    # 1. Predictions
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    
    # 2. Compute Metrics
    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred)
    rec = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    roc_auc = roc_auc_score(y_eval, y_prob)
    
    print("=" * 60)
    print(f"MODEL PERFORMANCE METRICS ({dataset_name.upper()})")
    print("=" * 60)
    print(f"Accuracy:         {acc:.4f}")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall:           {rec:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print("=" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_eval, y_pred))
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
