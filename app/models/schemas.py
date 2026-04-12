from pydantic import BaseModel
from typing import List


class SymptomResponse(BaseModel):
    symptoms: dict[str, str]


class PredictionRequest(BaseModel):
    symptoms: List[str]


class DiseaseSummary(BaseModel):
    disease: str
    summary: str
    source: str


class PredictionItem(BaseModel):
    disease: str
    probability: float
    percentage: float
    summary: DiseaseSummary


class PredictionResponse(BaseModel):
    selected_symptoms: list[str]
    symptom_count: int
    predicted_disease: str
    confidence: float
    confidence_percentage: float
    top_predictions: list[PredictionItem]


class VisualizationSummary(BaseModel):
    total_cases: int
    total_diseases: int
    total_symptoms: int
    average_symptoms_per_case: float


class TopSymptom(BaseModel):
    symptom: str
    count: int


class CorrelationHeatmapItem(BaseModel):
    xSymptom: str
    ySymptom: str
    correlation: float


class SymptomCountDistributionItem(BaseModel):
    symptomCount: int
    cases: int


class VisualizationData(BaseModel):
    summary: VisualizationSummary
    top_symptoms: list[TopSymptom]
    heatmap_symptoms: list[str]
    symptom_correlation_heatmap: list[CorrelationHeatmapItem]
    symptom_count_distribution: list[SymptomCountDistributionItem]


class VisualizationResponse(BaseModel):
    data: VisualizationData
    status: str = "success"


class ConfidenceDistributionItem(BaseModel):
    rangeLabel: str
    count: int
    rangeStart: float
    rangeEnd: float


class ModelConfidenceSummary(BaseModel):
    average_confidence: float
    median_confidence: float
    max_confidence: float
    top_one_accuracy: float
    top_three_accuracy: float
    test_case_count: int


class ModelConfidenceData(BaseModel):
    summary: ModelConfidenceSummary
    confidence_distribution: list[ConfidenceDistributionItem]


class ModelConfidenceResponse(BaseModel):
    data: ModelConfidenceData
    status: str = "success"
