# app/api/routes.py
import time
import numpy as np
import pandas as pd
from typing import Optional, List, Any, Tuple
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from app import globals
from app.services.soil import preprocess_image, get_weather_data, get_soil_type, get_location_details, geocode_location_from_text 
from app.services.fertilizer import predict_fertilizer, find_fertilizer_recommendation, generate_explanation, get_top_n_crop_suggestions, SOIL_CLASSES
from app.models import SoilPredictionResponse, CropSuggestion, FertilizerRecommendation, WeatherData

router = APIRouter()

SOIL_MAPPING = {name: i for i, name in enumerate(SOIL_CLASSES)}
TOP_N_CROPS = 3 

@router.get("/geocode")
async def geocode_location(state: str = Query(...), region: str = Query(...)):
    """
    Geocodes a text location (state, region) into latitude and longitude.
    Used by the 'Get Weather' (manual) button.
    """
    location_coords = geocode_location_from_text(state, region)
    if location_coords['latitude'] == 0.0 and location_coords['longitude'] == 0.0:
        raise HTTPException(status_code=400, detail="Location could not be resolved.")
    
    return location_coords

@router.get("/weather", response_model=WeatherData)
async def get_current_weather(lat: float = Query(...), lng: float = Query(...)):
    """
    Fetches weather details (temp, humidity, moisture) and reverse geocoding (country, state) 
    for a given latitude and longitude. Used by both location buttons.
    """
    weather_data = get_weather_data(lat, lng)
    location_details = get_location_details(lat, lng)
    
    return WeatherData(
        temperature=weather_data['temperature'],
        humidity=weather_data['humidity'],
        moisture=weather_data['moisture'],
        country=location_details['country'],
        state=location_details['state']
    )

@router.post("/predict/soil", response_model=SoilPredictionResponse)
async def predict_soil(
    image: Optional[UploadFile] = File(None), 
    state: str = Form(None),
    region: str = Form(None),
    lat: float = Form(None),
    lng: float = Form(None),
    nitrogen: Optional[int] = Form(None),
    phosphorous: Optional[int] = Form(None),
    potassium: Optional[int] = Form(None)
):
    start_time = time.time()
    
    if globals.soil_model is None or globals.crop_model is None or globals.fert_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded/trained. Please try again later.")

    location_provided = state or region or (lat is not None and lng is not None)
    npk_provided = nitrogen is not None or phosphorous is not None or potassium is not None
    image_provided = image is not None
    
    if not location_provided:
        raise HTTPException(status_code=400, detail="Location is required.")
    if not image_provided and not npk_provided:
        raise HTTPException(status_code=400, detail="Either a soil image or NPK values are required.")

    # Location Resolution
    location_coords = None
    location_details = None
    
    if lat is not None and lng is not None:
        location_coords = {'latitude': lat, 'longitude': lng}
        location_details = get_location_details(lat, lng)
    elif state and region:
        location_coords = geocode_location_from_text(state, region)
        location_details = get_location_details(location_coords['latitude'], location_coords['longitude'])


    if location_coords is None or location_details is None:
        print("ERROR: Final check failed - Location could not be resolved.")
        raise HTTPException(status_code=400, detail="Location could not be resolved.")
        
    # Soil Prediction (uses KNN model)
    predicted_soil_type = "Unknown Soil"
    
    if image_provided:
        try:
            processed_image = preprocess_image(image)
            predicted_soil_type = get_soil_type(globals.soil_model, processed_image)
        except Exception as e:
            print(f"Soil image processing failed, using default 'Unknown Soil': {e}")
            
    # CRITICAL FIX: VALIDATE SOIL TYPE BEFORE ML CALLS
    valid_soil_types = [s.lower() for s in SOIL_CLASSES]
    
    # Strip whitespace from the predicted type
    cleaned_soil_type = predicted_soil_type.strip()
    
    if cleaned_soil_type == "Unknown Soil":
        if not npk_provided:
             raise HTTPException(status_code=400, detail="Soil Type could not be determined and NPK values were not provided. Both are required for prediction.")
        
        raise HTTPException(status_code=400, detail="Soil Type could not be determined. Please ensure the image is clear or provide a valid combination of NPK values.")

    if cleaned_soil_type.lower() not in valid_soil_types: 
        raise HTTPException(status_code=500, detail=f"Invalid Soil Type '{predicted_soil_type}' predicted by model. Check model labels.")
            

    weather_data = get_weather_data(location_coords['latitude'], location_coords['longitude'])
    
    input_features = {
        'Temperature': weather_data['temperature'],
        'Humidity': weather_data['humidity'],
        'Moisture': weather_data['moisture'], 
        'Soil Type': cleaned_soil_type,
        
        'Soil_pH': 6.5, # Mock value for pH
        'Rainfall': 150.0, # Mock value for Rainfall (mm)
        'Nitrogen': nitrogen if nitrogen is not None else 0,
        'Phosphorous': phosphorous if phosphorous is not None else 0,
        'Potassium': potassium if potassium is not None else 0,
    }

    crop_input_features = {k: input_features[k] for k in [
        'Temperature', 
        'Humidity', 
        'Moisture', 
        'Soil Type', 
        'Soil_pH', 
        'Rainfall'
    ]}
    
    try:
        # Pass only the valid subset of features
        crop_suggestions = get_top_n_crop_suggestions(globals.crop_model, globals.crop_encoders, crop_input_features, TOP_N_CROPS)
    except Exception as e:
        print(f"ERROR: Crop prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Crop prediction failed: {e}")

    if not crop_suggestions:
        print("ERROR: Could not generate crop suggestions (empty list).")
        raise HTTPException(status_code=500, detail="Could not generate crop suggestions.")
        
    top_crop = crop_suggestions[0].name
    
    input_features['Crop Type'] = top_crop 
    try:
        predicted_fertilizer_name = predict_fertilizer(globals.fert_model, globals.fert_encoders, input_features)
    except Exception as e:
        print(f"ERROR: Fertilizer prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fertilizer prediction failed: {e}")
    
    # 3. Final steps
    recommendations = find_fertilizer_recommendation(globals.fertilizer_data, top_crop, predicted_fertilizer_name)
    explanation = generate_explanation(cleaned_soil_type, top_crop, predicted_fertilizer_name)
    inference_time = (time.time() - start_time) * 1000
    
    return SoilPredictionResponse(
        crop_suggestions=crop_suggestions, 
        fertilizer_recommendations=recommendations,
        explanation=explanation,
        weather_data=WeatherData(
            temperature=weather_data['temperature'],
            humidity=weather_data['humidity'],
            moisture=weather_data['moisture'],
            country=location_details['country'],
            state=location_details['state']
        ),
        inference_time_ms=inference_time,
        model_version="2.0 (Sklearn)"
    )