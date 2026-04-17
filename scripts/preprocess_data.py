from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "diseases_symptoms.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_DATA_PATH)

df.columns = (
    df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
)

X = df.drop("diseases", axis=1)
y = df["diseases"]

X.to_csv(PROCESSED_DATA_DIR / "X.csv", index=False)
y.to_csv(PROCESSED_DATA_DIR / "y.csv", index=False)

print("Done preprocessing!")
