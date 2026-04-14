import pandas as pd

df = pd.read_csv("data/raw/diseases_symptoms.csv")

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
print(df["diseases"].value_counts())

columns = [col for col in df.columns if col != "diseases"]
columns = sorted(columns)

symptoms = {col.lower().replace(" ", "_"): col.capitalize() for col in columns}

with open("app/ml/symptoms_map.json", "w") as f:
    import json

    json.dump(symptoms, f, indent=2)
