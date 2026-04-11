import os
import json
from fastapi import APIRouter, HTTPException

from app.models.schemas import SymptomResponse, PredictionRequest, PredictionResponse, VisualizationResponse, ModelConfidenceResponse
from app.services.predictor import model_service
from app.services.visualizer import get_model_confidence_data, get_visualization_data

router = APIRouter(prefix="/api")

@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/model-confidence")
def get_model_confidence():
    return ModelConfidenceResponse(data=get_model_confidence_data(), status="success")


@router.get("/visualizations")
def get_visualizations():
    return VisualizationResponse(data=get_visualization_data(), status="success")


@router.get("/symptoms")
def get_symptoms():
    with open(
        os.path.join(os.path.dirname(__file__), "..", "ml", "symptoms_map.json"), "r"
    ) as f:
        symptoms = json.load(f)
    return SymptomResponse(symptoms=symptoms)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    selected = list(dict.fromkeys(request.symptoms))
    features = model_service.get_features()

    if not selected:
        raise HTTPException(status_code=400, detail="At least one symptom is required.")

    unknown = [s for s in selected if s not in features]
    if unknown:
        raise HTTPException(status_code=400, detail={"unknown_symptoms": unknown})

    predictions = model_service.predict_from_symptoms(selected)
    response = PredictionResponse(
        selected_symptoms=selected,
        symptom_count=len(selected),
        predicted_disease=predictions[0]["disease"],
        confidence=predictions[0]["percentage"],
        confidence_percentage=predictions[0]["percentage"],
        top_predictions=predictions,
    )
    return response
