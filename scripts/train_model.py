import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("LR Accuracy:", accuracy_score(y_test, y_pred))

probabilities = model.predict_proba(X_train)
top_probabilities = probabilities.max(axis=1)
mean_probability = top_probabilities.mean()

print("Mean Probability:", mean_probability)

joblib.dump(model, "app/ml/lr_model.pkl")

# model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=None,
#     min_samples_split=2,
#     random_state=42,
#     n_jobs=-1
# )
# model.fit(X_train, y_train)

# print("RF Accuracy:", model.score(X_test, y_test))

# joblib.dump(model, "app/ml/rf_model.pkl")

print("Done training!")