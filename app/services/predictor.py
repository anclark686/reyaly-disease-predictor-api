from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "lr_model.pkl"
FEATURES_PATH = BASE_DIR / "ml" / "features.pkl"


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH, features_path: Path = FEATURES_PATH):
        self.model_path = model_path
        self.features_path = features_path
        self.model = joblib.load(self.model_path)
        self.features = joblib.load(self.features_path)

    def get_features(self) -> list[str]:
        return list(self.features)

    def get_disease_summary(self, disease: str) -> dict:
        formatted_name = disease.strip().replace(" ", "_").split("(")[0]
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_name}"

        try:
            response = requests.get(
                wiki_url,
                headers={"User-Agent": "MedicalApp/1.0"},
                timeout=5,
            )
        except requests.RequestException:
            response = None

        if response is None or response.status_code != 200:
            return {
                "disease": disease,
                "summary": f"https://www.google.com/search?q={quote_plus(disease)}",
                "source": "Google",
            }

        data = response.json()
        return {
            "disease": disease,
            "summary": data.get("extract"),
            "source": "Wikipedia",
        }

    def predict_from_symptoms(self, selected_symptoms: list[str]) -> list[dict]:
        input_data = {feature: 0 for feature in self.features}

        for symptom in selected_symptoms:
            if symptom in input_data:
                input_data[symptom] = 1

        input_df = pd.DataFrame([input_data], columns=self.features)

        probabilities = self.model.predict_proba(input_df)[0]
        classes = self.model.classes_

        ranked = sorted(
            [
                {
                    "disease": disease,
                    "probability": float(prob),
                    "percentage": float(f"{prob * 100:.2f}"),
                }
                for disease, prob in zip(classes, probabilities)
            ],
            key=lambda x: x["percentage"],
            reverse=True,
        )

        top_three = ranked[:3]

        for item in top_three:
            item["summary"] = self.get_disease_summary(item["disease"])

        return top_three


model_service = ModelService()
