# -*- coding: utf-8 -*-
"""
Auto-fix Crop_Recommendation.csv to add realistic, diverse values so the model
can actually separate crops (instead of collapsing to 5 clusters).

- Normalizes column names to: crop, temperature, humidity, moisture, soil ph, rainfall, soil type
- Standardizes crop names
- Assigns realistic Soil Type per crop (no ints, always string)
- Regenerates numeric features using realistic agronomic ranges + jitter
- Removes duplicates, shuffles, and writes a fixed CSV alongside the original
"""

import os
import sys
import math
import random
import pandas as pd
import numpy as np

RNG = np.random.default_rng(42)

# Where your CSV lives inside the container
CSV_IN  = "/app/backend/data/Crop_Recommendation.csv"
CSV_OUT = "/app/backend/data/Crop_Recommendation.fixed.csv"
CSV_BAK = "/app/backend/data/Crop_Recommendation.backup.csv"

# Canonical crop names we’ll keep
CROPS = [
    "Rice","Wheat","Maize","Cotton","Sugarcane","Potato","Tomato","Soybean",
    "Barley","Groundnut","Millet","Oilseed","Pulses","Jute","Coffee","Tea",
    "Coconut","Banana","Grapes","Apple",
]

# Friendly mapping from variants -> canonical crop name
CROP_NAME_MAP = {c.lower(): c for c in CROPS}

# Realistic soil choices per crop
SOIL_PER_CROP = {
    "Rice":      ["Clay","Loamy","Alluvial"],
    "Wheat":     ["Loamy","Clay"],
    "Maize":     ["Loamy","Alluvial","Red"],
    "Cotton":    ["Black","Red"],
    "Sugarcane": ["Loamy","Alluvial","Clay"],
    "Potato":    ["Loamy","Alluvial"],
    "Tomato":    ["Loamy","Alluvial","Red"],
    "Soybean":   ["Loamy","Black","Red"],
    "Barley":    ["Loamy","Clay"],
    "Groundnut": ["Sandy","Red","Loamy"],
    "Millet":    ["Sandy","Red","Loamy"],
    "Oilseed":   ["Loamy","Black","Red"],
    "Pulses":    ["Loamy","Alluvial","Black"],
    "Jute":      ["Alluvial","Loamy"],
    "Coffee":    ["Red","Loamy"],
    "Tea":       ["Red","Loamy"],
    "Coconut":   ["Sandy","Alluvial","Loamy"],
    "Banana":    ["Alluvial","Loamy"],
    "Grapes":    ["Black","Red","Loamy"],
    "Apple":     ["Alluvial","Loamy"],
}

# Typical agronomic ranges (min, max) per feature for each crop
# temperature(°C), humidity(%), moisture(%), soil ph, rainfall(mm)
RANGES = {
    "Rice":      {"temperature":(24,35),"humidity":(70,90),"moisture":(70,90),"soil ph":(5.5,7.0),"rainfall":(150,300)},
    "Wheat":     {"temperature":(10,25),"humidity":(50,70),"moisture":(40,60),"soil ph":(6.0,7.5),"rainfall":(75,125)},
    "Maize":     {"temperature":(20,32),"humidity":(50,70),"moisture":(50,70),"soil ph":(5.5,7.0),"rainfall":(75,150)},
    "Cotton":    {"temperature":(25,35),"humidity":(40,60),"moisture":(30,50),"soil ph":(6.0,8.0),"rainfall":(50,100)},
    "Sugarcane": {"temperature":(22,30),"humidity":(65,85),"moisture":(65,85),"soil ph":(6.0,7.5),"rainfall":(150,250)},
    "Potato":    {"temperature":(12,20),"humidity":(60,80),"moisture":(50,70),"soil ph":(5.0,6.5),"rainfall":(50,100)},
    "Tomato":    {"temperature":(20,28),"humidity":(50,70),"moisture":(50,70),"soil ph":(6.0,7.5),"rainfall":(50,100)},
    "Soybean":   {"temperature":(20,30),"humidity":(50,70),"moisture":(40,60),"soil ph":(6.0,7.5),"rainfall":(70,120)},
    "Barley":    {"temperature":(10,25),"humidity":(45,65),"moisture":(40,60),"soil ph":(6.0,7.5),"rainfall":(50,100)},
    "Groundnut": {"temperature":(22,30),"humidity":(50,70),"moisture":(35,55),"soil ph":(6.0,7.5),"rainfall":(50,100)},
    "Millet":    {"temperature":(25,35),"humidity":(40,60),"moisture":(30,50),"soil ph":(5.5,7.5),"rainfall":(40,80)},
    "Oilseed":   {"temperature":(18,30),"humidity":(45,65),"moisture":(40,60),"soil ph":(6.0,7.5),"rainfall":(50,120)},
    "Pulses":    {"temperature":(18,30),"humidity":(45,65),"moisture":(40,60),"soil ph":(6.0,7.5),"rainfall":(50,100)},
    "Jute":      {"temperature":(24,34),"humidity":(70,90),"moisture":(70,90),"soil ph":(6.0,7.5),"rainfall":(120,200)},
    "Coffee":    {"temperature":(18,28),"humidity":(60,80),"moisture":(55,75),"soil ph":(5.0,6.5),"rainfall":(120,200)},
    "Tea":       {"temperature":(18,25),"humidity":(70,90),"moisture":(60,80),"soil ph":(4.5,6.0),"rainfall":(150,300)},
    "Coconut":   {"temperature":(22,30),"humidity":(70,90),"moisture":(60,80),"soil ph":(5.5,7.5),"rainfall":(150,250)},
    "Banana":    {"temperature":(24,30),"humidity":(70,90),"moisture":(65,85),"soil ph":(5.5,7.5),"rainfall":(120,250)},
    "Grapes":    {"temperature":(15,30),"humidity":(40,60),"moisture":(40,60),"soil ph":(6.0,7.5),"rainfall":(50,100)},
    "Apple":     {"temperature":(10,20),"humidity":(50,70),"moisture":(50,70),"soil ph":(5.5,6.5),"rainfall":(50,100)},
}

EXPECTED = ["crop","temperature","humidity","moisture","soil ph","rainfall","soil type"]

def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lower-case names, drop unnamed, squish spaces
    mapper = {c: c.strip().lower() for c in df.columns if not str(c).lower().startswith("unnamed")}
    df = df.rename(columns=mapper)
    # unify common variants
    rename_map = {
        "label":"crop", "target":"crop", "soil_ph":"soil ph", "soil_pH":"soil ph",
        "soil ph value":"soil ph", "soil type ":"soil type", "soil_type":"soil type",
        "rain fall":"rainfall", "rain_fall":"rainfall",
        "temp":"temperature", "humid":"humidity",
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    # ensure required columns exist
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}. Found: {list(df.columns)}")
    return df[EXPECTED].copy()

def _canon_crop(x: str) -> str | None:
    if not isinstance(x,str): return None
    key = x.strip().lower()
    return CROP_NAME_MAP.get(key)

def _sample_uniform(a,b, size=None):
    return RNG.uniform(a,b, size)

def _jitter(val, pct=0.05):
    # add ±pct noise and clip to reasonable bounds later
    return val * (1 + RNG.normal(0, pct))

def _sample_feature(crop: str, feature: str, jitter_pct=0.10):
    lo, hi = RANGES[crop][feature]
    base = _sample_uniform(lo, hi)
    return float(np.clip(_jitter(base, jitter_pct), lo, hi))

def _fix_row(row):
    crop = row["crop"]
    # Soil type: keep string, pick valid for crop
    soil_choices = SOIL_PER_CROP.get(crop, ["Loamy"])
    row["soil type"] = str(RNG.choice(soil_choices))
    # Numeric features from realistic ranges
    row["temperature"] = _sample_feature(crop, "temperature")
    row["humidity"]    = _sample_feature(crop, "humidity")
    row["moisture"]    = _sample_feature(crop, "moisture")
    row["soil ph"]     = round(_sample_feature(crop, "soil ph"), 2)
    row["rainfall"]    = round(_sample_feature(crop, "rainfall"), 1)
    return row

def main():
    if not os.path.exists(CSV_IN):
        raise FileNotFoundError(f"Input CSV not found at {CSV_IN}")

    df = pd.read_csv(CSV_IN)
    df = _coerce_columns(df)

    # Standardize crop names and drop unknowns
    df["crop"] = df["crop"].apply(_canon_crop)
    before = len(df)
    df = df.dropna(subset=["crop"]).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before-after} rows with unknown crop labels.")

    # Generate realistic values per row
    df = df.apply(_fix_row, axis=1)

    # Ensure correct dtypes (critical: soil type must remain string)
    df["crop"] = df["crop"].astype(str)
    df["soil type"] = df["soil type"].astype(str)
    for col in ["temperature","humidity","moisture","soil ph","rainfall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicates that might exist
    df = df.drop_duplicates().reset_index(drop=True)

    # Shuffle for good measure
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Backup original once
    try:
        if not os.path.exists(CSV_BAK):
            os.link(CSV_IN, CSV_BAK) if hasattr(os, "link") else df.to_csv(CSV_BAK, index=False)
            print(f"Backed up original -> {CSV_BAK}")
    except Exception:
        # Windows/overlayfs may fail hard-link; ignore
        pass

    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"✅ Wrote fixed dataset -> {CSV_OUT}")
    print("Columns:", list(df.columns))
    print("Counts per crop:")
    print(df["crop"].value_counts().sort_index())

if __name__ == "__main__":
    main()
