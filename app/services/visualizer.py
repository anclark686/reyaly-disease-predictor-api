import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "diseases_symptoms.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "app" / "ml" / "lr_model.pkl"
ARTIFACTS_DIR = BASE_DIR / "app" / "ml" / "artifacts"
VISUALIZATION_ARTIFACT_PATH = ARTIFACTS_DIR / "visualizations.json"
MODEL_CONFIDENCE_ARTIFACT_PATH = ARTIFACTS_DIR / "model_confidence.json"


def _load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(PROCESSED_DATA_DIR / "X.csv")
    y = pd.read_csv(PROCESSED_DATA_DIR / "y.csv").squeeze()
    return X, y


def _write_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_artifact(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_visualization_data() -> dict:
    return _read_artifact(VISUALIZATION_ARTIFACT_PATH)


@lru_cache(maxsize=1)
def get_model_confidence_data() -> dict:
    return _read_artifact(MODEL_CONFIDENCE_ARTIFACT_PATH)


def build_visualization_data() -> dict:
    df = _load_dataset()

    symptom_columns = [column for column in df.columns if column != "diseases"]
    symptom_frame = df[symptom_columns]

    top_symptoms_series = symptom_frame.sum().sort_values(ascending=False).head(15)
    top_symptoms = [
        {"symptom": symptom, "count": int(count)}
        for symptom, count in top_symptoms_series.items()
    ]

    heatmap_symptoms = list(top_symptoms_series.head(10).index)
    correlation_matrix = symptom_frame[heatmap_symptoms].corr()
    symptom_correlation_heatmap = [
        {
            "xSymptom": x_symptom,
            "ySymptom": y_symptom,
            "correlation": round(
                float(correlation_matrix.loc[y_symptom, x_symptom]), 2
            ),
        }
        for y_symptom in heatmap_symptoms
        for x_symptom in heatmap_symptoms
    ]

    symptom_count_per_case = symptom_frame.sum(axis=1)
    symptom_count_distribution_series = (
        symptom_count_per_case.value_counts().sort_index()
    )
    symptom_count_distribution = [
        {"symptomCount": int(symptom_count), "cases": int(case_count)}
        for symptom_count, case_count in symptom_count_distribution_series.items()
    ]

    return {
        "summary": {
            "total_cases": int(len(df)),
            "total_diseases": int(df["diseases"].nunique()),
            "total_symptoms": int(len(symptom_columns)),
            "average_symptoms_per_case": round(float(symptom_count_per_case.mean()), 2),
        },
        "top_symptoms": top_symptoms,
        "heatmap_symptoms": heatmap_symptoms,
        "symptom_correlation_heatmap": symptom_correlation_heatmap,
        "symptom_count_distribution": symptom_count_distribution,
    }


def build_model_confidence_data() -> dict:
    X, y = _load_processed_data()
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)
    top_confidences = probabilities.max(axis=1) * 100
    top_predictions = model.predict(X_test)
    top_one_accuracy = accuracy_score(y_test, top_predictions) * 100

    class_labels = model.classes_
    top_three_indices = probabilities.argsort(axis=1)[:, -3:]
    top_three_predictions = class_labels[top_three_indices]
    top_three_matches = [
        actual_label in predicted_labels
        for actual_label, predicted_labels in zip(y_test, top_three_predictions)
    ]
    top_three_accuracy = (sum(top_three_matches) / len(top_three_matches)) * 100

    bin_edges = list(range(0, 101, 5))
    confidence_bins = pd.Series(
        pd.cut(
            top_confidences,
            bins=bin_edges,
            include_lowest=True,
            right=False,
        )
    )
    histogram = confidence_bins.value_counts(sort=False)

    confidence_distribution = [
        {
            "rangeLabel": f"{int(interval.left)}-{int(interval.right)}%",
            "count": int(count),
            "rangeStart": int(interval.left),
            "rangeEnd": int(interval.right),
        }
        for interval, count in histogram.items()
    ]

    return {
        "summary": {
            "average_confidence": round(float(top_confidences.mean()), 2),
            "median_confidence": round(float(pd.Series(top_confidences).median()), 2),
            "max_confidence": round(float(top_confidences.max()), 2),
            "top_one_accuracy": round(float(top_one_accuracy), 2),
            "top_three_accuracy": round(float(top_three_accuracy), 2),
            "test_case_count": int(len(top_confidences)),
        },
        "confidence_distribution": confidence_distribution,
    }


def generate_visualization_artifacts() -> None:
    _write_artifact(VISUALIZATION_ARTIFACT_PATH, build_visualization_data())
    _write_artifact(MODEL_CONFIDENCE_ARTIFACT_PATH, build_model_confidence_data())
