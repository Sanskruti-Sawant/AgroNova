# app/services/fertilizer.py - FIXED VERSION WITH DEBUGGING
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from app.models import CropSuggestion 
from sklearn.ensemble import RandomForestClassifier

# Define constants based on expected data
SOIL_CLASSES = ['Alluvial', 'Black', 'Clay', 'Red', 'Loamy', 'Sandy'] 
SOIL_CASE_MAP = {name.lower(): name for name in SOIL_CLASSES}

# Crop name standardization map (handles case variations)
CROP_NAME_MAP = {
    'rice': 'Rice',
    'wheat': 'Wheat',
    'maize': 'Maize',
    'cotton': 'Cotton',
    'sugarcane': 'Sugarcane',
    'potato': 'Potato',
    'tomato': 'Tomato',
    'soybean': 'Soybean',
    'barley': 'Barley',
    'groundnut': 'Groundnut',
    'millet': 'Millet',
    'oilseed': 'Oilseed',
    'pulses': 'Pulses',
    'jute': 'Jute',
    'coffee': 'Coffee',
    'tea': 'Tea',
    'coconut': 'Coconut',
    'banana': 'Banana',
    'grapes': 'Grapes',
    'apple': 'Apple',
}

# Define the features the Crop Model was trained on
CROP_MODEL_FEATURES = [
    'Temperature', 
    'Humidity', 
    'Moisture', 
    'Soil_pH', 
    'Rainfall', 
    'Soil Type'
]

# Define the features the Fertilizer Model was trained on
FERTILIZER_MODEL_FEATURES = [
    'Temperature', 
    'Humidity', 
    'Moisture', 
    'Soil Type', 
    'Crop Type', 
    'Nitrogen', 
    'Potassium', 
    'Phosphorous'
]


def get_top_n_crop_suggestions(
    rf_model: RandomForestClassifier, 
    encoders: Dict[str, Any], 
    input_features: Dict[str, Any], 
    n: int = 3
) -> List[CropSuggestion]:
    """
    Predicts crop probabilities using the Sklearn RF model and returns top N suggestions.
    
    CRITICAL FIXES:
    1. Extensive input validation
    2. Detailed debugging output
    3. Check for model bias
    """
    print(f"\n{'='*80}")
    print("CROP PREDICTION DEBUG")
    print(f"{'='*80}")
    print(f"Input features received: {input_features}")
    
    # Validate input features exist
    required_features = CROP_MODEL_FEATURES
    for feature in required_features:
        if feature not in input_features:
            print(f"❌ ERROR: Missing required feature '{feature}' in input")
            return []
    
    # Validate and convert numeric inputs
    try:
        temp = float(input_features['Temperature'])
        humidity = float(input_features['Humidity'])
        moisture = float(input_features['Moisture'])
        ph = float(input_features['Soil_pH'])
        rainfall = float(input_features['Rainfall'])
        
        print(f"\nNumeric inputs:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Moisture: {moisture}%")
        print(f"  pH: {ph}")
        print(f"  Rainfall: {rainfall}mm")
        
        # Sanity checks with warnings
        if not (0 <= temp <= 50):
            print(f"⚠️  WARNING: Temperature {temp}°C is outside typical range (0-50)")
        if not (0 <= humidity <= 100):
            print(f"⚠️  WARNING: Humidity {humidity}% is outside valid range (0-100)")
        if not (0 <= moisture <= 100):
            print(f"⚠️  WARNING: Moisture {moisture}% is outside valid range (0-100)")
        if not (0 <= ph <= 14):
            print(f"⚠️  WARNING: pH {ph} is outside valid range (0-14)")
        if not (0 <= rainfall <= 500):
            print(f"⚠️  WARNING: Rainfall {rainfall}mm is outside typical range (0-500)")
            
    except (ValueError, TypeError) as e:
        print(f"❌ ERROR: Invalid numeric input: {e}")
        return []
    
    # Encode Soil Type
    try:
        soil_type_input = str(input_features['Soil Type']).strip()
        standardized_soil_type = SOIL_CASE_MAP.get(soil_type_input.lower())
        
        if standardized_soil_type is None:
            print(f"❌ ERROR: Soil type '{soil_type_input}' not recognized.")
            print(f"   Valid types: {SOIL_CLASSES}")
            
            # Try to find closest match
            for key in SOIL_CASE_MAP.keys():
                if key in soil_type_input.lower():
                    standardized_soil_type = SOIL_CASE_MAP[key]
                    print(f"   INFO: Using closest match: {standardized_soil_type}")
                    break
            
            if standardized_soil_type is None:
                return []
            
        encoded_soil_type = encoders['Soil Type'].transform([standardized_soil_type])[0]
        print(f"\nSoil Type encoding:")
        print(f"  Input: '{soil_type_input}'")
        print(f"  Standardized: '{standardized_soil_type}'")
        print(f"  Encoded: {encoded_soil_type}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to encode Soil Type: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Prepare input data with EXACT feature order
    input_data = {
        'Temperature': temp,
        'Humidity': humidity,
        'Moisture': moisture,
        'Soil_pH': ph,
        'Rainfall': rainfall,
        'Soil Type': encoded_soil_type
    }
    
    # Create DataFrame with explicit column order
    input_df = pd.DataFrame([input_data], columns=CROP_MODEL_FEATURES)
    
    print(f"\nInput DataFrame for prediction:")
    print(input_df)
    print(f"\nDataFrame dtypes:")
    print(input_df.dtypes)
    
    # Get probabilities
    try:
        probabilities = rf_model.predict_proba(input_df)[0]
        class_labels = rf_model.classes_
        
        print(f"\n{'='*80}")
        print("PREDICTION RESULTS")
        print(f"{'='*80}")
        print(f"Total classes in model: {len(class_labels)}")
        print(f"Probabilities computed: {len(probabilities)}")
        
        # Log ALL predictions to check for bias
        all_predictions = sorted(
            zip(class_labels, probabilities), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\nTop 10 Crop Predictions:")
        for i, (label, prob) in enumerate(all_predictions[:10], 1):
            try:
                crop_name = encoders['Crop Type'].inverse_transform([label])[0]
                bar = "█" * int(prob * 50)
                print(f"  {i:2}. {crop_name:15} {bar} {prob:.4f} ({prob*100:.2f}%)")
            except Exception:
                print(f"  {i:2}. [Unknown Crop {label}]: {prob:.4f}")
        
        # Check for prediction bias (if one crop dominates)
        max_prob = max(probabilities)
        if max_prob > 0.9:
            print(f"\n⚠️  WARNING: Very high confidence ({max_prob:.2%}) - possible model bias!")
        
        # Check diversity of predictions
        significant_preds = [p for p in probabilities if p > 0.01]
        print(f"\nPrediction diversity: {len(significant_preds)} crops with >1% probability")
        
        if len(significant_preds) < 5:
            print(f"⚠️  WARNING: Low prediction diversity - model may be biased!")
        
        # Get top N indices
        top_n_indices = np.argsort(probabilities)[::-1][:n]
        suggestions = []
        
        print(f"\n{'='*80}")
        print(f"FINAL TOP {n} SUGGESTIONS")
        print(f"{'='*80}")
        
        # Inverse transform to crop names
        for index in top_n_indices:
            encoded_label = class_labels[index]
            try:
                crop_name = encoders['Crop Type'].inverse_transform([encoded_label])[0]
                confidence = float(probabilities[index])
                suggestions.append(CropSuggestion(name=crop_name, confidence=confidence))
                print(f"✓ {crop_name}: {confidence:.4f} ({confidence*100:.2f}%)")
            except Exception as e:
                print(f"❌ ERROR: Failed to decode crop label {encoded_label}: {e}")
        
        print(f"{'='*80}\n")
                
    except Exception as e:
        print(f"❌ ERROR: Crop prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return []
            
    return suggestions


def predict_fertilizer(
    fert_model: RandomForestClassifier, 
    fert_encoders: Dict[str, Any], 
    input_features: Dict[str, Any]
) -> str:
    """
    Predicts the best fertilizer type using the Sklearn RF model and encoders.
    """
    
    print(f"\n{'='*80}")
    print("FERTILIZER PREDICTION DEBUG")
    print(f"{'='*80}")
    print(f"Input features: {input_features}")
    
    # Validate required features
    required_features = FERTILIZER_MODEL_FEATURES
    
    missing = [f for f in required_features if f not in input_features]
    if missing:
        print(f"❌ ERROR: Missing required features: {missing}")
        return "Unknown Fertilizer - Missing Input Data"
    
    # Validate numeric inputs
    try:
        temp = float(input_features['Temperature'])
        humidity = float(input_features['Humidity'])
        moisture = float(input_features['Moisture'])
        nitrogen = float(input_features['Nitrogen'])
        potassium = float(input_features['Potassium'])
        phosphorous = float(input_features['Phosphorous'])
        
        print(f"Numeric values: T={temp}, H={humidity}, M={moisture}, N={nitrogen}, K={potassium}, P={phosphorous}")
        
    except (ValueError, TypeError) as e:
        print(f"❌ ERROR: Invalid numeric input: {e}")
        return "Unknown Fertilizer - Invalid Numeric Input"
    
    # Encode Soil Type
    try:
        soil_type_input = str(input_features['Soil Type']).strip()
        standardized_soil_type = SOIL_CASE_MAP.get(soil_type_input.lower())
        
        if standardized_soil_type is None:
            print(f"❌ ERROR: Soil type '{soil_type_input}' not recognized")
            return "Unknown Fertilizer - Invalid Soil Type"
        
        if standardized_soil_type not in fert_encoders['Soil Type'].classes_:
            print(f"❌ ERROR: Soil type '{standardized_soil_type}' not in fertilizer training data")
            return "Unknown Fertilizer - Soil Type Not in Training Data"
            
        encoded_soil = fert_encoders['Soil Type'].transform([standardized_soil_type])[0]
        print(f"Soil Type: '{soil_type_input}' -> '{standardized_soil_type}' -> {encoded_soil}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to encode Soil Type: {e}")
        return "Unknown Fertilizer - Soil Encoding Error"
    
    # Encode Crop Type with standardization
    try:
        crop_type_input = str(input_features['Crop Type']).strip()
        standardized_crop = CROP_NAME_MAP.get(crop_type_input.lower(), crop_type_input)
        
        print(f"Crop Type: '{crop_type_input}' -> standardized: '{standardized_crop}'")
        
        if standardized_crop not in fert_encoders['Crop Type'].classes_:
            print(f"⚠️  WARNING: Crop '{standardized_crop}' not in fertilizer training data")
            
            # Try exact case match
            available_crops = list(fert_encoders['Crop Type'].classes_)
            for crop in available_crops:
                if crop.lower() == crop_type_input.lower():
                    standardized_crop = crop
                    print(f"   Found exact case match: {standardized_crop}")
                    break
            else:
                print(f"❌ ERROR: No match found for crop '{crop_type_input}'")
                return f"Unknown Fertilizer - Crop '{crop_type_input}' Not Supported"
        
        encoded_crop = fert_encoders['Crop Type'].transform([standardized_crop])[0]
        print(f"Crop Type encoded: '{standardized_crop}' -> {encoded_crop}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to encode Crop Type: {e}")
        return "Unknown Fertilizer - Crop Encoding Error"
    
    # Prepare input with explicit feature order
    input_data = {
        'Temperature': temp,
        'Humidity': humidity,
        'Moisture': moisture,
        'Soil Type': encoded_soil,
        'Crop Type': encoded_crop,
        'Nitrogen': nitrogen,
        'Potassium': potassium,
        'Phosphorous': phosphorous
    }
    
    input_df = pd.DataFrame([input_data], columns=FERTILIZER_MODEL_FEATURES)
    print(f"\nFertilizer Input DataFrame:\n{input_df}")
    
    # Predict fertilizer
    try:
        pred_encoded = fert_model.predict(input_df)[0]
        fertilizer_name = fert_encoders['Fertilizer Name'].inverse_transform([pred_encoded])[0]
        print(f"✓ Predicted fertilizer: {fertilizer_name}")
        print(f"{'='*80}\n")
        return fertilizer_name
        
    except Exception as e:
        print(f"❌ ERROR: Fertilizer prediction failed: {e}")
        print(f"{'='*80}\n")
        return "Unknown Fertilizer - Prediction Failed"


def find_fertilizer_recommendation(
    fertilizer_data: pd.DataFrame, 
    crop_type: str, 
    fertilizer_name: str
) -> List[Dict[str, str]]:
    """Provides a human-readable recommendation from the dataset."""
    recommendations = []
    
    print(f"\nSearching recommendations for: Crop='{crop_type}', Fertilizer='{fertilizer_name}'")
    
    # Case-insensitive matching
    filtered_data = fertilizer_data[
        (fertilizer_data['Crop Type'].str.strip().str.lower() == crop_type.strip().lower()) & 
        (fertilizer_data['Fertilizer Name'].str.strip().str.lower() == fertilizer_name.strip().lower())
    ]
    
    if not filtered_data.empty:
        row = filtered_data.iloc[0]
        rec = {
            "name": row['Fertilizer Name'],
            "amount": "200 kg per acre", 
            "frequency": "Once per season",
        }
        recommendations.append(rec)
        print(f"✓ Found recommendation: {rec}")
    else:
        print(f"⚠️  WARNING: No specific recommendation found")
        recommendations.append({
            "name": fertilizer_name,
            "amount": "Follow standard agricultural guidelines",
            "frequency": "Varies by crop cycle",
        })
        
    return recommendations


def generate_explanation(soil_type: str, crop_type: str, fertilizer_name: str) -> str:
    """Generates a human-readable explanation for the prediction."""
    explanation = (
        f"Based on our analysis, your soil is primarily **{soil_type}**. "
        f"This type of soil is well-suited for growing **{crop_type}**. "
        f"The recommended fertilizer, **{fertilizer_name}**, is suggested to provide "
        f"the necessary nutrients for a healthy crop yield."
    )
    return explanation