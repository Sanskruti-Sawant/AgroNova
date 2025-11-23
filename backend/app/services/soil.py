# app/services/soil.py
import os
import io
import json
import math
import time
import base64
import logging
from typing import Dict, Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

if not OWM_KEY:
    logger.warning("OPENWEATHER_API_KEY not found in environment. Weather & geocoding will fail with 400.")
OWM_GEO_URL = "https://api.openweathermap.org/geo/1.0"
OWM_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

HTTP_TIMEOUT = 12  
SESSION = requests.Session()


def preprocess_image(upload_file) -> np.ndarray:
    """
    Convert uploaded file to the exact feature vector shape your KNN expects.

    The model error you saw (expects 4096 features) means it was trained on 64x64.
    We resize to 64x64, convert to grayscale, flatten to length 4096, then scale to [0,1].
    """
    try:
        content = upload_file.file.read()
        img = Image.open(io.BytesIO(content)).convert("L")  
        img = img.resize((64, 64))                          
        arr = np.asarray(img, dtype=np.float32) / 255.0
        vec = arr.flatten()                               
        return vec
    finally:
        try:
            upload_file.file.seek(0)
        except Exception:
            pass


#Geocoding(STATE + REGION -> lat/lon)
def geocode_location_from_text(state: str, region: str) -> Dict[str, float]:
    """
    Use OpenWeather Direct Geocoding (requires API key).
    Example query: q="Amritsar,Punjab,IN"
    """
    if not OWM_KEY:
        return {"latitude": 0.0, "longitude": 0.0}

    q = f"{region},{state},IN"
    params = {"q": q, "limit": 1, "appid": OWM_KEY}
    r = SESSION.get(f"{OWM_GEO_URL}/direct", params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data:
        return {"latitude": 0.0, "longitude": 0.0}
    first = data[0]
    return {"latitude": float(first["lat"]), "longitude": float(first["lon"])}


def get_location_details(lat: float, lng: float) -> Dict[str, str]:
    """
    Use OpenWeather Reverse Geocoding.
    """
    if not OWM_KEY:
        return {"country": "Unknown", "state": "Unknown"}

    params = {"lat": lat, "lon": lng, "limit": 1, "appid": OWM_KEY}
    r = SESSION.get(f"{OWM_GEO_URL}/reverse", params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data:
        return {"country": "Unknown", "state": "Unknown"}

    first = data[0]
    country = first.get("country", "Unknown")
    state = first.get("state") or first.get("name") or "Unknown"
    return {"country": country, "state": state}

def get_weather_data(lat: float, lng: float) -> Dict[str, float]:
    """
    Fetch current weather using OpenWeather /weather endpoint.
    Returns temperature (°C), humidity (%), and a proxy 'moisture' (%).
    """
    if not OWM_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY missing; cannot fetch weather.")

    params = {"lat": lat, "lon": lng, "appid": OWM_KEY, "units": "metric"}
    r = SESSION.get(OWM_WEATHER_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    main = data.get("main", {})
    temp = float(main.get("temp", 0.0))
    humidity = float(main.get("humidity", 0.0))

    moisture = max(20.0, min(90.0, humidity * 0.9))

    return {
        "temperature": temp,
        "humidity": humidity,
        "moisture": moisture,
    }


def get_soil_type(knn_model, feature_vec: np.ndarray) -> str:
    """
    Predict soil class from preprocessed vector.
    """
    pred = knn_model.predict([feature_vec])[0]
    return str(pred)
