# app/services/ml_utils.py
from __future__ import annotations

import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

def _norm_col(c: str) -> str:
    return (
        str(c).strip()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
        .lower()
    )

# 1) SOIL IMAGE PIPELINE (KNN)
#    Directory layout expected:
#      soil_dataset/
#        Alluvial/  *.jpg|*.png...
#        Black/     ...
#        Clay/      ...
#        Red/       ...
#        Loamy/     ...
#        Sandy/     ...
def load_soil_images(
    dataset_dir: str,
    image_size: Tuple[int, int] = (64, 64)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load soil images from class subfolders and return (X, y).
    Uses Pillow to avoid OpenCV dependency issues.
    """
    from PIL import Image

    X: List[np.ndarray] = []
    y: List[str] = []

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Soil dataset directory not found: {dataset_dir}")

    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert("L")  
                img = img.resize(image_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
                X.append(arr.flatten())
                y.append(class_name)
            except Exception:
                continue

    if not X:
        raise RuntimeError(f"No images loaded from {dataset_dir}. Check folder names and files.")

    X_arr = np.vstack(X)
    y_arr = np.array(y)
    return X_arr, y_arr


def train_knn_classifier(X: np.ndarray, y: np.ndarray):
    """Simple KNN classifier for soil type."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
    knn.fit(X, y)
    return knn


from typing import Tuple

def load_and_prepare_crop_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Loads the crop dataset and returns features/target + metadata.
    Auto-detects target/feature column name variants and normalizes them.
    """
    df = pd.read_csv(csv_path)

    norm_map = {c: _norm_col(c) for c in df.columns}
    df = df.rename(columns=norm_map)

    target_aliases = ["label", "crop", "crop name", "crop type", "target"]
    target_col = next((c for c in target_aliases if c in df.columns), None)
    if target_col is None:
        raise ValueError(
            f"Missing target column. Looked for one of: {target_aliases}. "
            f"Found columns: {list(df.columns)}"
        )

    alias_map = {
        "temperature": ["temperature", "temp"],
        "humidity":    ["humidity", "humid"],
        "moisture":    ["moisture", "soil moisture", "soilmoisture"],
        "soil ph":     ["soil ph", "ph", "soil_pH".lower()],
        "rainfall":    ["rainfall", "rain fall", "rain"],
        "soil type":   ["soil type", "soil", "soiltype", "soil_type"],
    }

    resolved = {}
    for canonical, candidates in alias_map.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            resolved[canonical] = found

    required = ["temperature", "humidity", "moisture", "soil ph", "rainfall", "soil type"]
    missing = [c for c in required if c not in resolved]
    if missing:
        raise ValueError(
            "Missing required feature columns. "
            f"Need {required}. Could not find: {missing}. "
            f"Available: {list(df.columns)}"
        )

    X = pd.DataFrame({
        "Temperature": df[resolved["temperature"]],
        "Humidity":    df[resolved["humidity"]],
        "Moisture":    df[resolved["moisture"]],
        "Soil_pH":     df[resolved["soil ph"]],
        "Rainfall":    df[resolved["rainfall"]],
        "Soil Type":   df[resolved["soil type"]].astype(str),
    })

    y = df[target_col].astype(str)

    meta = {
        "num_cols": ["Temperature", "Humidity", "Moisture", "Soil_pH", "Rainfall"],
        "cat_cols": ["Soil Type"],
        "target_col": "Label", 
        "original_target_col": target_col,
    }

    print(f"\nColumn mapping:")
    print(f"  target -> '{target_col}'")
    for k, v in resolved.items():
        print(f"  {k} -> '{v}'")

    return X, y, meta


def train_crop_model(X: pd.DataFrame, y: pd.Series, meta: Dict[str, Any]):
    """
    Train a calibrated Random Forest with proper preprocessing (no leakage),
    optional oversampling on the TRAIN split, and return:
        model, encoders
      - model: CalibratedClassifierCV wrapping Pipeline(preproc -> RandomForest)
      - encoders: {'Soil Type': <dummy with .classes_>, 'pipeline': preproc}
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_train_bal, y_train_bal = X_train, y_train
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
        print(f"✅ Oversampled training set to {len(X_train_bal)} samples (balanced).")
    except Exception:
        print("ℹ️  imblearn not available; skipping oversampling.")

    base_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        min_samples_split=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    rf_pipe = Pipeline(steps=[("preproc", preproc), ("clf", base_rf)])

    calib = CalibratedClassifierCV(
        estimator=rf_pipe,
        method="isotonic",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    )

    print("\nTraining calibrated Random Forest...")
    calib.fit(X_train_bal, y_train_bal)
    print("✅ Training complete.")

    y_pred = calib.predict(X_test)
    print("\nValidation report (held-out test):")
    print(classification_report(y_test, y_pred, digits=3))

    soil_classes = sorted(pd.Series(X["Soil Type"]).astype(str).unique())

    class _DummyEncoder:
        def __init__(self, classes):
            self.classes_ = np.array(classes)

    soil_encoder = _DummyEncoder(soil_classes)

    encoders = {
        "Soil Type": soil_encoder,
        "pipeline": preproc,
    }

    return calib, encoders


def load_and_prepare_fertilizer_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load fertilizer dataset and return encoded X, y and encoders.
    Expected (with aliases handled):
      Inputs: Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
      Target: Fertilizer Name (or similar)
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_path)

    norm_map = {c: _norm_col(c) for c in df.columns}
    df = df.rename(columns=norm_map)

    alias_map = {
        "temperature": ["temperature", "temp"],
        "humidity":    ["humidity", "humid"],
        "moisture":    ["moisture", "soil moisture", "soilmoisture"],
        "soil type":   ["soil type", "soil", "soiltype", "soil_type"],
        "crop type":   ["crop type", "crop", "crop name"],
        "nitrogen":    ["nitrogen", "n"],
        "potassium":   ["potassium", "k"],
        "phosphorous": ["phosphorous", "phosphorus", "p"],
        "fertilizer name": ["fertilizer name", "fertilizer", "fertiliser name"],
    }

    resolved = {}
    for canonical, candidates in alias_map.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            resolved[canonical] = found

    required = [
        "temperature", "humidity", "moisture",
        "soil type", "crop type",
        "nitrogen", "potassium", "phosphorous",
        "fertilizer name",
    ]
    missing = [c for c in required if c not in resolved]
    if missing:
        raise ValueError(
            f"Fertilizer CSV missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    X_raw = pd.DataFrame({
        "Temperature": df[resolved["temperature"]],
        "Humidity":    df[resolved["humidity"]],
        "Moisture":    df[resolved["moisture"]],
        "Soil Type":   df[resolved["soil type"]].astype(str),
        "Crop Type":   df[resolved["crop type"]].astype(str),
        "Nitrogen":    df[resolved["nitrogen"]],
        "Potassium":   df[resolved["potassium"]],
        "Phosphorous": df[resolved["phosphorous"]],
    })

    y_raw = df[resolved["fertilizer name"]].astype(str)

    soil_le = LabelEncoder().fit(X_raw["Soil Type"])
    crop_le = LabelEncoder().fit(X_raw["Crop Type"])
    fert_le = LabelEncoder().fit(y_raw)

    X_enc = X_raw.copy()
    X_enc["Soil Type"] = soil_le.transform(X_raw["Soil Type"])
    X_enc["Crop Type"] = crop_le.transform(X_raw["Crop Type"])

    y_enc = fert_le.transform(y_raw)

    encoders = {
        "Soil Type": soil_le,
        "Crop Type": crop_le,
        "Fertilizer Name": fert_le,
    }

    return X_enc, pd.Series(y_enc), encoders


def train_fertilizer_model(X: pd.DataFrame, y: pd.Series):
    """Simple RF on already-encoded fertilizer features."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def predict_crop_topk(
    model,
    payload: Dict[str, Any],
    top_k: int = 3,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to get top-k crop predictions from your calibrated RF pipeline.
    Expects payload keys: Temperature, Humidity, Moisture, Soil_pH, Rainfall, Soil Type
    """
    import pandas as pd

    cols = ["Temperature", "Humidity", "Moisture", "Soil_pH", "Rainfall", "Soil Type"]
    row = {
        "Temperature": float(payload["Temperature"]),
        "Humidity": float(payload["Humidity"]),
        "Moisture": float(payload["Moisture"]),
        "Soil_pH": float(payload["Soil_pH"]),
        "Rainfall": float(payload["Rainfall"]),
        "Soil Type": str(payload["Soil Type"]),
    }
    X = pd.DataFrame([row], columns=cols)
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    order = np.argsort(probs)[::-1][:top_k]
    out = [{"crop": classes[i], "confidence": float(probs[i])} for i in order]
    if verbose:
        print("Top-k:", out)
    return out
