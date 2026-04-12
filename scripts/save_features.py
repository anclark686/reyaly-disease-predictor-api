from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
ML_DIR = BASE_DIR / "app" / "ml"

ML_DIR.mkdir(parents=True, exist_ok=True)

# Load the processed data
X = pd.read_csv(PROCESSED_DATA_DIR / "X.csv")

# Save the feature names
features = X.columns.tolist()
joblib.dump(features, ML_DIR / "features.pkl")

print(f"Saved {len(features)} features to app/ml/features.pkl")
