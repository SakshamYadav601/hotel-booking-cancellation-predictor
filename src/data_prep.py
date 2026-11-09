"""
data_prep.py
Responsible for dataset loading, structure exploration, initial data quality checks,
and descriptive statistics.
Lead: Documentation Lead (Saksham Yadav)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up project-relative paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"


def load_dataset(filepath=DATA_PATH) -> pd.DataFrame:
    """Loads the dataset from the project data directory using relative paths."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please verify path.")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded successfully from: {filepath}")
    return df


def explore_dataset_structure(df: pd.DataFrame) -> dict:
    """
    Explores dataset shape, column names, memory usage, and data types.
    Returns a dictionary summarizing structural properties.
    """
    structure_summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    
    print("=" * 60)
    print("DATASET STRUCTURE OVERVIEW")
    print("=" * 60)
    print(f"Total Rows:    {structure_summary['rows']}")
    print(f"Total Columns: {structure_summary['columns']}")
    print(f"Memory Usage:  {structure_summary['memory_mb']} MB")
    print("\nData Types Count:")
    print(df.dtypes.value_counts())
    print("=" * 60)
    
    return structure_summary


def perform_data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing values, percentage missing, unique values, and duplicate rows.
    Returns a summary DataFrame sorted by missing values count.
    """
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    unique_counts = df.nunique()
    
    quality_df = pd.DataFrame({
        "Missing_Count": missing_counts,
        "Missing_Percent": missing_percentages.round(2),
        "Unique_Values": unique_counts,
        "Data_Type": df.dtypes
    })
    
    # Filter for columns with missing values for concise reporting
    cols_with_na = quality_df[quality_df["Missing_Count"] > 0]
    duplicate_rows = df.duplicated().sum()
    
    print("\n" + "=" * 60)
    print("DATA QUALITY & INTEGRITY AUDIT")
    print("=" * 60)
    print(f"Total Duplicate Rows: {duplicate_rows}")
    print(f"Columns with Missing Values: {len(cols_with_na)}")
    if not cols_with_na.empty:
        print("\nColumns with Missing Values Detail:")
        print(cols_with_na[["Missing_Count", "Missing_Percent"]])
    else:
        print("No missing values detected.")
    print("=" * 60)
    
    return quality_df


def compute_descriptive_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates summary statistics (mean, median, std dev, min, max, quartiles)
    for numeric features and generates the correlation matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Summary statistics table
    stats_df = numeric_df.describe().T
    stats_df["median"] = numeric_df.median()
    stats_df = stats_df[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
    
    # Correlation matrix
    corr_matrix = numeric_df.corr().round(3)
    
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS (NUMERICAL FEATURES)")
    print("=" * 60)
    print(stats_df.head(10))
    print("=" * 60)
    
    return stats_df, corr_matrix


if __name__ == "__main__":
    # Test execution locally
    df_raw = load_dataset()
    explore_dataset_structure(df_raw)
    perform_data_quality_checks(df_raw)
    compute_descriptive_statistics(df_raw)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Cleans the raw dataframe:
    - Removes duplicate rows
    - Drops columns with excessive missing values (>90%)
    - Drops leakage columns (columns that reveal reservation status ahead of time)
    - Imputes missing values for agent, country, and children
    """
    df_clean = df.copy()
    
    # 1. Handle duplicate entries
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        
    # 2. Drop columns with >90% missing values
    if "company" in df_clean.columns:
        df_clean = df_clean.drop(columns=["company"])
        
    # 3. Drop data leakage columns that directly expose cancellation status
    leakage_cols = ["reservation_status", "reservation_status_date"]
    df_clean = df_clean.drop(columns=[col for col in leakage_cols if col in df_clean.columns])
    
    # 4. Impute missing values
    if "agent" in df_clean.columns:
        df_clean["agent"] = df_clean["agent"].fillna(0)
    if "country" in df_clean.columns:
        df_clean["country"] = df_clean["country"].fillna("Unknown")
    if "children" in df_clean.columns:
        df_clean["children"] = df_clean["children"].fillna(0)
        
    return df_clean


def preprocess_and_split(df: pd.DataFrame, target_col: str = "is_canceled"):
    """
    Encodes categorical variables, standardizes numerical features,
    and performs an 80/20 stratified train-test split.
    """
    # Clean the dataset first
    df_clean = clean_data(df)
    
    # Separate features (X) and target label (y)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # One-hot encode categorical features (convert strings to numeric dummy columns)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 80/20 Stratified train-test split (stratify preserves target class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling: fit scaler strictly on training set to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    print(f"[INFO] Preprocessing complete.")
    print(f"[INFO] Training set shape: {X_train_scaled.shape}")
    print(f"[INFO] Test set shape:     {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test