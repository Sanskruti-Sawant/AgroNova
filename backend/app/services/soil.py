# app/services/soil.py
import numpy as np
import os
import io
import requests
from typing import Dict, List, Any, Tuple
from fastapi import UploadFile
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from app import globals 
from app.models import WeatherData # Import necessary model for type hinting

IMAGE_SIZE = (64, 64) # Must match ml_utils training size

# --- Geocoding Function ---
def geocode_location_from_text(state: str, region: str) -> Dict[str, float]:
    """
    Converts state and region to lat/lng using OpenWeatherMap's Geo API.
    """
    # FIX: Read the environment variable inside the function
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY") 
    
    if not OPENWEATHER_API_KEY:
        print("WARNING: OPENWEATHER_API_KEY not found. Returning mock coordinates.")
        return {"latitude": 18.5204, "longitude": 73.8567}

    location_query = f"{region},{state},IN"
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_query}&limit=1&appid={OPENWEATHER_API_KEY}"
    
    try:
        # Increased timeout to 15s for stability
        response = requests.get(url, timeout=15) 
        response.raise_for_status() 
        data = response.json()
        
        if data:
            return {"latitude": data[0]['lat'], "longitude": data[0]['lon']}
        
        print(f"Geocoding failed for {location_query}. Empty response.")
    
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Geocoding API request failed: {e}")
        
    return {"latitude": 0.0, "longitude": 0.0}

# --- Reverse Geocoding Function ---
def get_location_details(lat: float, lon: float) -> Dict[str, str]:
    """
    Fetches country and state from lat/lng using OpenWeatherMap's reverse geocoding API.
    """
    # FIX: Read the environment variable inside the function
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
    
    if not OPENWEATHER_API_KEY:
        return {"country": "India", "state": "Maharashtra"}
        
    url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHER_API_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data:
            country = data[0].get('country', 'Unknown')
            state = data[0].get('state', 'Unknown')
            return {"country": country, "state": state}
        
        print("Location details failed. Empty response.")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Location Details API request failed: {e}")
        
    return {"country": "Unknown", "state": "Unknown"}

# --- Weather Data Function ---
def get_weather_data(lat: float, lon: float) -> Dict[str, float]:
    """
    Fetches current temperature and humidity based on location.
    Returns mock data on failure.
    """
    # FIX: Read the environment variable inside the function
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
    
    if not OPENWEATHER_API_KEY:
        print("WARNING: OPENWEATHER_API_KEY not found. Returning mock weather data.")
        return {"temperature": 25.0, "humidity": 60.0, "moisture": 40.0}
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Catches 4xx, 5xx errors
        data = response.json()
        
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        # Approximate soil moisture
        moisture = humidity * 0.7
        
        return {"temperature": temperature, "humidity": float(humidity), "moisture": moisture}
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Weather API request failed. Network/API key issue: {e}")
    except KeyError:
        print("ERROR: Weather API response missing expected 'main' data structure.")
    except Exception as e:
        print(f"ERROR: General weather data retrieval error: {e}")

    # Fallback to mock data if API call or processing fails
    return {"temperature": 25.0, "humidity": 60.0, "moisture": 40.0}
        
# --- Soil Prediction Functions ---

def preprocess_image(image_file: UploadFile) -> np.ndarray:
# ... (rest of the file remains the same)
    """
    Reads, resizes, flattens, and normalizes an uploaded image for the KNN model.
    """
    try:
        image_bytes = image_file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to read and open image file: {e}")

    image = image.resize(IMAGE_SIZE)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Flatten and normalize as expected by the KNN model
    img_array = np.array(image).flatten().reshape(1, -1)
    img_array = img_array.astype('float64') / 255.0
    
    return img_array

def get_soil_type(knn_model: KNeighborsClassifier, processed_image: np.ndarray) -> str:
    """
    Performs inference on the KNN soil model to predict soil type.
    """
    if knn_model is None:
         return "Unknown Soil"
         
    # KNN model predicts the label string directly
    return knn_model.predict(processed_image)[0]