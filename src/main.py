"""
main.py
Lead: Model Evaluation (Member 4)
Orchestrates model evaluation metrics, plot generations, and unseen data testing.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Imports from team scripts
from src.data_prep import preprocess_and_split  # Member 2
from src.model import train_logistic_regression  # Member 3
from src.plots import (  # Member 4
    plot_and_save_confusion_matrix,
    plot_and_save_correlation_heatmap,
    plot_and_save_roc_curve,
    test_unseen_sample,
)


def main():
    print("=== STARTING HOTEL CANCELLATION PIPELINE ===")

    # 1. Load Raw Data & Get Splits
    data_path = "data/hotel_bookings.csv"
    print(f"[INFO] Loading {data_path}...")
    df_raw = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = preprocess_and_split(df_raw)

    # 2. Train Model
    print("[INFO] Training Logistic Regression model...")
    model = train_logistic_regression(X_train, y_train)


    print("\n[INFO] Evaluating model performance...")

    # A. Generate Predictions & Probabilities
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # B. Compute Evaluation Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_prob)

    # C. Print Metrics Summary Report
    print("\nModel Evaluation Metrics:")
    print("------------------------------------------")
    print(f"Training Set Accuracy : {train_acc:.4f}")
    print(f"Testing Set Accuracy  : {test_acc:.4f}")
    print(f"Testing Set Precision : {test_prec:.4f}")
    print(f"Testing Set Recall    : {test_rec:.4f}")
    print(f"Testing Set F1-Score  : {test_f1:.4f}")
    print(f"Testing Set ROC-AUC   : {test_auc:.4f}")
    print("------------------------------------------\n")

    # D. Save Visual Figures (Correlation Heatmap, Confusion Matrix, ROC Curve)
    plot_and_save_correlation_heatmap(
        df_raw, output_path="outputs/correlation_heatmap.png"
    )
    plot_and_save_confusion_matrix(
        model, X_test, y_test, output_path="outputs/confusion_matrix.png"
    )
    plot_and_save_roc_curve(
        model, X_test, y_test, output_path="outputs/roc_curve.png"
    )

    # E. Test Model on an Unseen Sample
    unseen_sample = df_raw.drop(columns=["is_canceled"]).iloc[[0]]
    test_unseen_sample(model, unseen_sample, expected_columns=X_train.columns.tolist())

    print("Pipeline executed successfully.")
  

if __name__ == "__main__":
    main()
