# backend/app/main.py
import os
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import globals
from app.api.routes import router
from app.services.ml_utils import (
    load_soil_images,
    train_knn_classifier,
    load_and_prepare_crop_data,
    train_crop_model,
    load_and_prepare_fertilizer_data,
    train_fertilizer_model,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Training Sklearn models and loading data...")

    repo_root = Path(__file__).resolve().parent.parent
    base_path = repo_root / "data"

    soil_dataset_dir = base_path / "soil_dataset"
    fertilizer_csv_path = base_path / "Fertilizer Prediction.csv"
    crop_csv_path = base_path / "Crop_Recommendation.csv"

    try:
        print("Training Soil Model (KNN)...")
        X_soil, y_soil = load_soil_images(str(soil_dataset_dir))
        globals.soil_model = train_knn_classifier(X_soil, y_soil)

        print("Training Crop Model (RF)...")
        X_crop, y_crop, crop_meta = load_and_prepare_crop_data(str(crop_csv_path))
        crop_model, crop_encoders = train_crop_model(X_crop, y_crop, crop_meta)
        globals.crop_model = crop_model
        globals.crop_encoders = crop_encoders

        print("Training Fertilizer Model (RF)...")
        X_fert, y_fert, fert_encoders = load_and_prepare_fertilizer_data(str(fertilizer_csv_path))
        globals.fert_model = train_fertilizer_model(X_fert, y_fert)
        globals.fert_encoders = fert_encoders
        globals.fertilizer_data = pd.read_csv(fertilizer_csv_path)

        print("Models and data loaded/trained successfully!")
    except Exception as e:
        print(f"Error loading/training models or data: {e}")
        raise RuntimeError(f"Failed to load/train models at startup: {e}") from e

    yield


app = FastAPI(
    title="Agronova API (Sklearn)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Agronova Sklearn API is up and running!"}