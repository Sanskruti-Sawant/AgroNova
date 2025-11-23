import os
import pandas as pd
from backend.app.services.ml_utils import (
    load_and_prepare_crop_data,
    train_crop_model,
    predict_crop
)

from collections import defaultdict, Counter

# Test input sets for each climate regime (5 regimes × 6 soil types = 30 tests)
CLIMATE_REGIMES = [
    ("Hot & Humid (Rice-like)", 28, 75, 85, 6.8, 220),
    ("Cool & Moderate (Wheat-like)", 20, 60, 50, 6.5, 100),
    ("Hot & Dry (Cotton-like)", 33, 50, 35, 7.6, 75),
    ("Moderate (Maize-like)", 29, 66, 62, 6.1, 125),
    ("Cool & Humid (Potato-like)", 18, 65, 55, 5.5, 80),
]

SOIL_TYPES = ["Clay", "Loamy", "Sandy", "Black", "Red", "Alluvial"]

EXPECTED_PRIMARY = {
    "Hot & Humid (Rice-like)": "Rice",
    "Cool & Moderate (Wheat-like)": "Wheat",
    "Hot & Dry (Cotton-like)": "Cotton",
    "Moderate (Maize-like)": "Maize",
    "Cool & Humid (Potato-like)": "Potato",
}


def test_crop_model_with_diverse_inputs(model, encoders):
    print("\n" + "="*80)
    print("TESTING MODEL WITH DIVERSE INPUTS")
    print("="*80)

    results = []

    for regime_name, t, h, m, ph, r in CLIMATE_REGIMES:
        print("\n" + "─" * 80)
        print(f"Testing Regime: {regime_name}")
        print("─" * 80)

        for soil in SOIL_TYPES:
            features = {
                "Temperature": t,
                "Humidity": h,
                "Moisture": m,
                "Soil_pH": ph,
                "Rainfall": r,
                "Soil Type": soil,
            }

            print("\n" + "="*80)
            print("CROP PREDICTION DEBUG")
            print("="*80)
            print(f"Input features received: {features}")

            top3 = predict_crop(model, encoders, features, top_k=3)
            top1 = top3[0][0]

            print("\nFINAL TOP 3 SUGGESTIONS")
            print("="*80)
            for name, prob in top3:
                print(f"✓ {name}: {prob:.4f} ({prob*100:.2f}%)")
            print("="*80)

            results.append({
                "regime": regime_name,
                "soil": soil,
                "top1": top1,
                "top3": [x[0] for x in top3]
            })

    return results


def main():
    print("\n" + "#"*80)
    print("CROP MODEL DIAGNOSTIC TOOL")
    print("#"*80 + "\n")

    csv_path = "/app/backend/data/Crop_Recommendation.csv"
    print(f"Loading data from: {csv_path}\n")

    X, y, encoders = load_and_prepare_crop_data(csv_path)
    model, meta = train_crop_model(X, y)

    results = test_crop_model_with_diverse_inputs(model, encoders)

    print("\n" + "="*80)
    print("PREDICTION QUALITY CHECKS")
    print("="*80)

    by_regime = defaultdict(list)
    for r in results:
        by_regime[r["regime"]].append(r)

    all_ok = True
    for regime, rows in by_regime.items():
        expected = EXPECTED_PRIMARY.get(regime)
        hits = sum(1 for r in rows if r["top1"] == expected)
        total = len(rows)
        ok = hits == total
        all_ok = all_ok and ok
        print(f"{'✓' if ok else '✗'} {regime}: {hits}/{total} match expected → {expected}")

    print("\n" + "="*80)
    print("DIVERSITY SUMMARY (informational)")
    print("="*80)

    total = len(results)
    unique_crops = set(r["top1"] for r in results)
    print(f"Total predictions: {total}")
    print(f"Unique crops predicted: {len(unique_crops)} ({unique_crops})")
    print("Note: Low diversity here is expected because climate dominates choices.\n")

    print("#"*80)
    if all_ok:
        print("✓ DIAGNOSTIC PASSED - Model predicts correct primary crops per regime")
    else:
        print("✗ DIAGNOSTIC FAILED - At least one regime misclassified")
    print("#"*80)


if __name__ == "__main__":
    main()
