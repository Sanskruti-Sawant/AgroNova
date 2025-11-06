"""
Diagnostic script to test crop model predictions and identify bias.
Run this AFTER training your models to verify they work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml_utils import load_and_prepare_crop_data, train_crop_model
from app.services.fertilizer import get_top_n_crop_suggestions
import pandas as pd
import numpy as np


def test_crop_model_with_diverse_inputs(model, encoders):
    """
    Tests the crop model with a variety of different inputs to check
    if it predicts different crops or is biased toward one.
    """
    print(f"\n{'='*80}")
    print("TESTING MODEL WITH DIVERSE INPUTS")
    print(f"{'='*80}")
    
    soil_encoder = encoders['Soil Type']
    
    # Define diverse test cases that should produce different crop predictions
    test_cases = [
        {
            "name": "Hot & Humid (Rice-like)",
            "temp": 28, "humidity": 75, "moisture": 85, "ph": 6.8, "rainfall": 220,
            "expected_crops": ["Rice", "Sugarcane", "Jute"]
        },
        {
            "name": "Cool & Moderate (Wheat-like)",
            "temp": 20, "humidity": 60, "moisture": 50, "ph": 6.5, "rainfall": 100,
            "expected_crops": ["Wheat", "Barley", "Potato"]
        },
        {
            "name": "Hot & Dry (Cotton-like)",
            "temp": 33, "humidity": 50, "moisture": 35, "ph": 7.6, "rainfall": 75,
            "expected_crops": ["Cotton", "Oilseed", "Groundnut"]
        },
        {
            "name": "Moderate (Maize-like)",
            "temp": 29, "humidity": 66, "moisture": 62, "ph": 6.1, "rainfall": 125,
            "expected_crops": ["Maize", "Soybean", "Millet"]
        },
        {
            "name": "Cool & Humid (Potato-like)",
            "temp": 18, "humidity": 65, "moisture": 55, "ph": 5.5, "rainfall": 80,
            "expected_crops": ["Potato", "Barley", "Apple"]
        },
    ]
    
    soil_types = ['Clay', 'Loamy', 'Sandy', 'Black', 'Red', 'Alluvial']
    
    predictions_by_soil = {}
    all_predictions = []
    
    for soil_type in soil_types:
        if soil_type not in soil_encoder.classes_:
            print(f"⚠️  Skipping '{soil_type}' - not in encoder classes")
            continue
            
        predictions_by_soil[soil_type] = []
        
        print(f"\n{'─'*80}")
        print(f"Testing with Soil Type: {soil_type}")
        print(f"{'─'*80}")
        
        for test in test_cases:
            input_features = {
                'Temperature': test['temp'],
                'Humidity': test['humidity'],
                'Moisture': test['moisture'],
                'Soil_pH': test['ph'],
                'Rainfall': test['rainfall'],
                'Soil Type': soil_type
            }
            
            suggestions = get_top_n_crop_suggestions(model, encoders, input_features, n=3)
            
            if suggestions:
                top_crop = suggestions[0].name
                confidence = suggestions[0].confidence
                predictions_by_soil[soil_type].append(top_crop)
                all_predictions.append(top_crop)
                
                # Check if prediction is in expected crops
                match = "✓" if top_crop in test['expected_crops'] else "✗"
                
                print(f"  {match} {test['name']:30} -> {top_crop:15} ({confidence:.3f})")
                print(f"      Top 3: {', '.join([f'{s.name}({s.confidence:.2f})' for s in suggestions])}")
                print(f"      Expected: {', '.join(test['expected_crops'])}")
            else:
                print(f"  ❌ {test['name']:30} -> PREDICTION FAILED")
    
    # Analyze prediction diversity
    print(f"\n{'='*80}")
    print("PREDICTION DIVERSITY ANALYSIS")
    print(f"{'='*80}")
    
    unique_predictions = set(all_predictions)
    total_predictions = len(all_predictions)
    
    print(f"Total predictions made: {total_predictions}")
    print(f"Unique crops predicted: {len(unique_predictions)}")
    print(f"Diversity ratio: {len(unique_predictions)}/{total_predictions} ({len(unique_predictions)/total_predictions*100:.1f}%)")
    
    # Count predictions
    from collections import Counter
    pred_counts = Counter(all_predictions)
    
    print(f"\nPrediction frequency:")
    for crop, count in pred_counts.most_common():
        pct = (count / total_predictions) * 100
        bar = "█" * int(pct / 2)
        print(f"  {crop:15} {bar} {count:3} times ({pct:5.1f}%)")
    
    # Check for bias
    most_common = pred_counts.most_common(1)[0]
    if most_common[1] / total_predictions > 0.5:
        print(f"\n⚠️  CRITICAL: MODEL IS BIASED!")
        print(f"    '{most_common[0]}' predicted {most_common[1]}/{total_predictions} times ({most_common[1]/total_predictions*100:.1f}%)")
        print(f"    The model is NOT learning to differentiate between crops properly.")
        return False
    elif len(unique_predictions) < 10:
        print(f"\n⚠️  WARNING: Low prediction diversity!")
        print(f"    Only {len(unique_predictions)} different crops predicted")
        print(f"    Model may still have bias issues")
        return False
    else:
        print(f"\n✓ Model shows good prediction diversity!")
        return True


def main():
    """Main diagnostic function."""
    print(f"\n{'#'*80}")
    print("CROP MODEL DIAGNOSTIC TOOL")
    print(f"{'#'*80}")
    
    # Load and prepare data
    csv_path = 'path/to/Crop_Recommendation.csv'  # UPDATE THIS PATH
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: CSV file not found at {csv_path}")
        print(f"   Please update the csv_path in this script to point to your CSV file.")
        return
    
    print(f"\nLoading data from: {csv_path}")
    
    try:
        X, y, encoders = load_and_prepare_crop_data(csv_path)
        
        # Train model
        model = train_crop_model(X, y)
        
        # Run diagnostics
        success = test_crop_model_with_diverse_inputs(model, encoders)
        
        if success:
            print(f"\n{'#'*80}")
            print("✓ DIAGNOSTIC PASSED - Model appears to be working correctly!")
            print(f"{'#'*80}\n")
        else:
            print(f"\n{'#'*80}")
            print("✗ DIAGNOSTIC FAILED - Model has prediction bias!")
            print(f"{'#'*80}")
            print("\nRECOMMENDATIONS:")
            print("1. Collect more training data (aim for 100+ samples per crop)")
            print("2. Ensure data is balanced across all crop types")
            print("3. Check for data quality issues (duplicates, errors)")
            print("4. Try different model hyperparameters")
            print("5. Consider using SMOTE or other balancing techniques")
            print(f"{'#'*80}\n")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()