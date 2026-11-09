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

