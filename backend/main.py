# main.py
import os
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import globals 
from app.api.routes import router
from app.services.ml_utils import (
    load_soil_images, train_knn_classifier, 
    load_and_prepare_crop_data, train_crop_model,
    load_and_prepare_fertilizer_data, train_fertilizer_model
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Training Sklearn models and loading data...")
    # Base path points to the 'backend/data' directory
    base_path = os.path.join(os.path.dirname(__file__), 'data')
    
    # Define file paths
    soil_dataset_dir = os.path.join(base_path, 'soil_dataset')
    fertilizer_csv_path = os.path.join(base_path, 'Fertilizer Prediction.csv')
    crop_csv_path = os.path.join(base_path, 'Crop_Recommendation.csv') 

    try:
        # 1. Train Soil Model (KNN)
        print("Training Soil Model (KNN)...")
        X_soil, y_soil = load_soil_images(soil_dataset_dir)
        globals.soil_model = train_knn_classifier(X_soil, y_soil)
        
        # 2. Train Crop Model (Random Forest)
        print("Training Crop Model (RF)...")
        X_crop, y_crop, globals.crop_encoders = load_and_prepare_crop_data(crop_csv_path)
        globals.crop_model = train_crop_model(X_crop, y_crop)

        # 3. Train Fertilizer Model (Random Forest)
        print("Training Fertilizer Model (RF)...")
        X_fert, y_fert, globals.fert_encoders = load_and_prepare_fertilizer_data(fertilizer_csv_path)
        globals.fert_model = train_fertilizer_model(X_fert, y_fert)
        globals.fertilizer_data = pd.read_csv(fertilizer_csv_path) 

        print("Models and data loaded/trained successfully!")
    except Exception as e:
        print(f"Error loading/training models or data: {e}")
        raise RuntimeError(f"Failed to load/train models at startup: {e}") from e
    
    yield

app = FastAPI(
    title="Agronova API (Sklearn)",
    version="2.0.0",
    lifespan=lifespan
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Agronova Sklearn API is up and running!"}