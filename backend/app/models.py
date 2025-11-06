# app/models.py
from pydantic import BaseModel
from typing import List, Optional

class WeatherData(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    country: str
    state: str

class CropSuggestion(BaseModel):
    name: str
    confidence: float

class FertilizerRecommendation(BaseModel):
    name: str
    amount: str
    frequency: str

class SoilPredictionResponse(BaseModel):
    crop_suggestions: List[CropSuggestion]
    fertilizer_recommendations: List[FertilizerRecommendation]
    explanation: str
    weather_data: WeatherData
    inference_time_ms: float
    model_version: str