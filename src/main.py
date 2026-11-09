"""
main.py
Master execution script for the hotel cancellation prediction pipeline.
"""

from src.data_prep import load_dataset, preprocess_and_split
from src.model import train_logistic_regression, evaluate_model
from src.plots import plot_and_save_visualizations


def main():
    print("=== STARTING HOTEL CANCELLATION PIPELINE ===")
    
    # 1. Data Preparation (Unpacks 4 values matching data_prep.py)
    df = load_dataset()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    
    # 2. Model Training
    model = train_logistic_regression(X_train, y_train)
    
    # 3. Model Evaluation & Visualizations
    test_metrics = evaluate_model(model, X_test, y_test, dataset_name="Test Set")
    plot_and_save_visualizations(model, X_test, y_test, prefix="test")
    
    print("\n=== PIPELINE EXECUTED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()