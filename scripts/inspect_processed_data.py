import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "diseases_symptoms.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "app" / "ml" / "lr_model.pkl"
ARTIFACTS_DIR = BASE_DIR / "app" / "ml" / "artifacts"
VISUALIZATION_ARTIFACT_PATH = ARTIFACTS_DIR / "visualizations.json"
MODEL_CONFIDENCE_ARTIFACT_PATH = ARTIFACTS_DIR / "model_confidence.json"


def _load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(PROCESSED_DATA_DIR / "X.csv")
    y = pd.read_csv(PROCESSED_DATA_DIR / "y.csv").squeeze()
    return X, y


def _load_dataset() -> pd.DataFrame:
    X, y = _load_processed_data()
    df = X.copy()
    df["diseases"] = y
    return df


def inspect_processed_data():
    df = _load_dataset()
    print("df.head()")
    print(df.head())
    print()
    print("df.shape")
    print(df.shape)
    print()
    print("df.columns")
    print(df.columns.tolist())
    print()
    print("df.isna().sum()")
    print(df.isna().sum())
    print()
    print("df.duplicated().sum()")
    print(df.duplicated().sum())
    print()
    print("df['diseases'].nunique()")
    print(df["diseases"].nunique())
    print()
    print("df['diseases'].value_counts()")
    for disease, count in df["diseases"].value_counts()[:100].items():
        print(f"{disease}: {count}")


if __name__ == "__main__":
    inspect_processed_data()
