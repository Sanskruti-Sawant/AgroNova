# app/services/fertilizer.py - FULL CORRECTED VERSION
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from app.models import CropSuggestion

# ---------------------------------------------------------------------
# Canonical classes / names
# ---------------------------------------------------------------------
SOIL_CLASSES = ["Alluvial", "Black", "Clay", "Red", "Loamy", "Sandy"]
SOIL_CASE_MAP = {s.lower(): s for s in SOIL_CLASSES}

CROP_NAME_MAP = {
    "rice": "Rice", "wheat": "Wheat", "maize": "Maize", "cotton": "Cotton",
    "sugarcane": "Sugarcane", "potato": "Potato", "tomato": "Tomato",
    "soybean": "Soybean", "barley": "Barley", "groundnut": "Groundnut",
    "millet": "Millet", "oilseed": "Oilseed", "pulses": "Pulses", "jute": "Jute",
    "coffee": "Coffee", "tea": "Tea", "coconut": "Coconut",
    "banana": "Banana", "grapes": "Grapes", "apple": "Apple",
}

# Features used by the crop (RF) pipeline
CROP_MODEL_FEATURES = [
    "Temperature", "Humidity", "Moisture", "Soil_pH", "Rainfall", "Soil Type",
]

# Features used by the fertilizer model
FERTILIZER_MODEL_FEATURES = [
    "Temperature", "Humidity", "Moisture", "Soil Type",
    "Crop Type", "Nitrogen", "Potassium", "Phosphorous",
]

# ---------------------------------------------------------------------
# Crop suggestions (top-N) using the calibrated RF + preprocessing pipeline
# IMPORTANT: We pass "Soil Type" as a STRING, the pipeline one-hot-encodes it.
# ---------------------------------------------------------------------
def get_top_n_crop_suggestions(
    rf_model: RandomForestClassifier,
    encoders: Dict[str, Any],
    input_features: Dict[str, Any],
    n: int = 3,
) -> List[CropSuggestion]:
    """
    Returns top-N crop suggestions with probabilities from the trained crop model.
    Expects input_features to include:
      Temperature, Humidity, Moisture, Soil_pH, Rainfall, Soil Type  (Soil Type as string)
    """
    print("\n" + "=" * 80)
    print("CROP PREDICTION DEBUG")
    print("=" * 80)
    print(f"Input features received: {input_features}")

    # Validate numeric
    try:
        temp = float(input_features["Temperature"])
        humidity = float(input_features["Humidity"])
        moisture = float(input_features["Moisture"])
        ph = float(input_features["Soil_pH"])
        rainfall = float(input_features["Rainfall"])
    except Exception as e:
        print(f"❌ ERROR: invalid numeric input: {e}")
        return []

    # Normalize soil type (keep as string for the pipeline)
    soil_raw = str(input_features["Soil Type"]).strip().lower()
    soil_value = SOIL_CASE_MAP.get(soil_raw)
    if soil_value is None:
        print(f"❌ ERROR: Soil type '{input_features['Soil Type']}' not recognized. Valid: {SOIL_CLASSES}")
        return []

    # Build input row in exact order
    input_df = pd.DataFrame(
        [{
            "Temperature": temp,
            "Humidity": humidity,
            "Moisture": moisture,
            "Soil_pH": ph,
            "Rainfall": rainfall,
            "Soil Type": soil_value,  # <- categorical STRING
        }],
        columns=CROP_MODEL_FEATURES,
    )

    print("\nInput DataFrame for prediction:")
    print(input_df)

    # Predict proba
    try:
        probabilities = rf_model.predict_proba(input_df)[0]
        class_labels = rf_model.classes_  # final crop names
    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        return []

    ranked = sorted(zip(class_labels, probabilities), key=lambda x: x[1], reverse=True)
    top = ranked[:n]

    print("\n" + "=" * 80)
    print(f"FINAL TOP {n} SUGGESTIONS")
    print("=" * 80)
    suggestions: List[CropSuggestion] = []
    for crop, prob in top:
        suggestions.append(CropSuggestion(name=crop, confidence=float(prob)))
        print(f"✓ {crop}: {prob:.4f} ({prob*100:.2f}%)")
    print("=" * 80 + "\n")

    return suggestions

# ---------------------------------------------------------------------
# Fertilizer prediction using prefit encoders
# ---------------------------------------------------------------------
def predict_fertilizer(
    fert_model: RandomForestClassifier,
    fert_encoders: Dict[str, Any],
    input_features: Dict[str, Any],
) -> str:
    """
    Predict fertilizer name given numeric + encoded categorical features.
    Requires:
      Temperature, Humidity, Moisture, Soil Type (string), Crop Type (string),
      Nitrogen, Potassium, Phosphorous
    """
    print("\n" + "=" * 80)
    print("FERTILIZER PREDICTION DEBUG")
    print("=" * 80)
    print(f"Input features: {input_features}")

    # Validate numeric
    try:
        temp = float(input_features["Temperature"])
        humidity = float(input_features["Humidity"])
        moisture = float(input_features["Moisture"])
        nitrogen = float(input_features["Nitrogen"])
        potassium = float(input_features["Potassium"])
        phosphorous = float(input_features["Phosphorous"])
    except Exception as e:
        print(f"❌ ERROR: invalid numeric input: {e}")
        return "Unknown Fertilizer - Invalid numeric input"

    # Encode soil
    soil_raw = str(input_features["Soil Type"]).strip().lower()
    soil_value = SOIL_CASE_MAP.get(soil_raw)
    if soil_value not in getattr(fert_encoders.get("Soil Type"), "classes_", []):
        return "Unknown Fertilizer - Soil Type not trained"
    encoded_soil = fert_encoders["Soil Type"].transform([soil_value])[0]

    # Encode crop
    raw_crop = str(input_features["Crop Type"]).strip().lower()
    crop_standard = CROP_NAME_MAP.get(raw_crop, input_features["Crop Type"])
    if crop_standard not in getattr(fert_encoders.get("Crop Type"), "classes_", []):
        return f"Unknown Fertilizer - Crop '{crop_standard}' not supported"
    encoded_crop = fert_encoders["Crop Type"].transform([crop_standard])[0]

    # Assemble input row
    input_df = pd.DataFrame(
        [{
            "Temperature": temp,
            "Humidity": humidity,
            "Moisture": moisture,
            "Soil Type": encoded_soil,
            "Crop Type": encoded_crop,
            "Nitrogen": nitrogen,
            "Potassium": potassium,
            "Phosphorous": phosphorous,
        }],
        columns=FERTILIZER_MODEL_FEATURES,
    )

    try:
        pred_encoded = fert_model.predict(input_df)[0]
        fertilizer_name = fert_encoders["Fertilizer Name"].inverse_transform([pred_encoded])[0]
        print(f"✓ Predicted Fertilizer: {fertilizer_name}")
        print("=" * 80 + "\n")
        return fertilizer_name
    except Exception as e:
        print(f"❌ ERROR: Fertilizer prediction failed: {e}")
        print("=" * 80 + "\n")
        return "Unknown Fertilizer - Prediction Failed"

# ---------------------------------------------------------------------
# Human-readable recommendation from dataset (simple look-up)
# ---------------------------------------------------------------------
def find_fertilizer_recommendation(
    fertilizer_data: pd.DataFrame,
    crop_type: str,
    fertilizer_name: str,
) -> List[Dict[str, str]]:
    """
    Returns a small, human-readable recommendation list from your fertilizer CSV.
    """
    recs: List[Dict[str, str]] = []
    print(f"\nSearching recommendations for: Crop='{crop_type}', Fertilizer='{fertilizer_name}'")

    try:
        df = fertilizer_data.copy()
        ccol = "Crop Type"
        fcol = "Fertilizer Name"
        if ccol not in df.columns or fcol not in df.columns:
            # Fallback if columns are named slightly differently
            df_cols = {c.lower(): c for c in df.columns}
            ccol = df_cols.get("crop type", ccol)
            fcol = df_cols.get("fertilizer name", fcol)

        match = df[
            (df[ccol].astype(str).str.strip().str.lower() == crop_type.strip().lower()) &
            (df[fcol].astype(str).str.strip().str.lower() == fertilizer_name.strip().lower())
        ]
        if not match.empty:
            row = match.iloc[0]
            recs.append({
                "name": str(row[fcol]),
                "amount": "200 kg per acre",
                "frequency": "Once per season",
            })
            print(f"✓ Found recommendation: {recs[-1]}")
        else:
            print("⚠️  No exact recommendation row found; returning generic.")
            recs.append({
                "name": fertilizer_name,
                "amount": "Follow standard agricultural guidelines",
                "frequency": "Varies by crop cycle",
            })
    except Exception as e:
        print(f"⚠️  Recommendation lookup failed: {e}")
        recs.append({
            "name": fertilizer_name,
            "amount": "Follow standard agricultural guidelines",
            "frequency": "Varies by crop cycle",
        })

    return recs

# ---------------------------------------------------------------------
# Human-friendly explanation
# ---------------------------------------------------------------------
def generate_explanation(soil_type: str, crop_type: str, fertilizer_name: str) -> str:
    return (
        f"Based on current conditions and detected **{soil_type}** soil, "
        f"the model suggests growing **{crop_type}**. "
        f"Recommended fertilizer: **{fertilizer_name}** to support nutrient needs."
    )

__all__ = [
    "SOIL_CLASSES",
    "get_top_n_crop_suggestions",
    "predict_fertilizer",
    "find_fertilizer_recommendation",
    "generate_explanation",
    "CROP_MODEL_FEATURES",
    "FERTILIZER_MODEL_FEATURES",
]
