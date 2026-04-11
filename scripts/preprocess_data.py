import pandas as pd

df = pd.read_csv("data/raw/diseases_symptoms.csv")

# clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# remove duplicates
df = df.drop_duplicates()

# split features/target
X = df.drop("diseases", axis=1)
y = df["diseases"]

# save cleaned data
X.to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)

print("Done preprocessing!")