import pandas as pd
import joblib

# Load the processed data
X = pd.read_csv("data/processed/X.csv")

# Save the feature names
features = X.columns.tolist()
joblib.dump(features, "app/ml/features.pkl")

print(f"Saved {len(features)} features to app/ml/features.pkl")
