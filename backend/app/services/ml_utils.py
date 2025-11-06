# app/services/ml_utils.py - FIXED VERSION
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Tuple

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to prevent KeyErrors and fix common typos."""
    df.columns = df.columns.str.strip()
    
    # Fix common typos
    if 'Temparature' in df.columns:
        df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    if 'Soil_Type' in df.columns:
        df.rename(columns={'Soil_Type': 'Soil Type'}, inplace=True)
    if 'Crop' in df.columns:
        df.rename(columns={'Crop': 'Crop Type'}, inplace=True)
        
    return df


# --- Soil Image Functions (KNN) ---

def load_soil_images(dataset_dir: str, image_size: Tuple[int, int] = (64, 64)):
    """Loads and flattens soil images from subfolders for training."""
    X, y = [], []
    
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Soil dataset directory not found at {dataset_dir}")
    
    labels = sorted(os.listdir(dataset_dir))  # Sort for consistency
    print(f"Loading soil images from: {dataset_dir}")
    print(f"Found soil classes: {labels}")
    
    for label in labels:
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
            
        image_count = 0
        for file in os.listdir(folder):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder, file)
                try:
                    img = Image.open(img_path).resize(image_size).convert('RGB')
                    X.append(np.array(img).flatten())
                    y.append(label)
                    image_count += 1
                except Exception as e:
                    print(f"WARNING: Failed to load {img_path}: {e}")
        
        print(f"  Loaded {image_count} images for class '{label}'")
    
    if not X:
        raise ValueError(f"No soil images found in {dataset_dir}. Check file structure.")
    
    print(f"Total images loaded: {len(X)}")
    return np.array(X) / 255.0, np.array(y)


def train_knn_classifier(X: np.ndarray, y: np.ndarray, k: int = 3):
    """Trains the K-Nearest Neighbors classifier."""
    print(f"Training KNN classifier with k={k}...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    print(f"KNN training complete. Classes: {knn.classes_}")
    return knn


# --- Fertilizer Prediction Functions (RF) ---

def load_and_prepare_fertilizer_data(csv_path: str):
    """
    Loads CSV, encodes categorical features, and prepares for RF training.
    """
    print(f"\nLoading fertilizer data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Initial columns: {df.columns.tolist()}")
    
    df = clean_df(df)
    print(f"After cleaning: {df.columns.tolist()}")
    
    # Validate required columns
    required_cols = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 
                     'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer Name']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in fertilizer data: {missing_cols}")
    
    print(f"Fertilizer data shape: {df.shape}")
    print(f"Unique Soil Types: {df['Soil Type'].unique()}")
    print(f"Unique Crop Types: {df['Crop Type'].nunique()} crops")
    print(f"Unique Fertilizers: {df['Fertilizer Name'].unique()}")
    
    # Encode categorical features
    encoders = {}
    for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    # Define explicit feature order
    feature_columns = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 
                      'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    
    X = df[feature_columns]
    y = df['Fertilizer Name']
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Feature order: {X.columns.tolist()}")
    
    return X, y, encoders


def train_fertilizer_model(X: pd.DataFrame, y: pd.Series):
    """Trains the Random Forest classifier for fertilizer."""
    print(f"\nTraining fertilizer model...")
    print(f"Training samples: {len(X)}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced'
    )
    
    model.fit(X, y)
    
    print(f"Fertilizer model training complete")
    print(f"Feature importances: {dict(zip(X.columns, model.feature_importances_))}")
    
    return model


# --- Crop Prediction Functions (RF) - CRITICAL FIXES ---

def load_and_prepare_crop_data(csv_path: str):
    """
    Loads crop recommendation CSV and prepares it for training.
    
    CRITICAL FIXES:
    1. Added data augmentation to handle small dataset
    2. Explicit validation and diagnostics
    3. Proper train/test split for validation
    """
    print(f"\n{'='*80}")
    print("LOADING CROP TRAINING DATA")
    print(f"{'='*80}")
    
    df = pd.read_csv(csv_path)
    print(f"Initial shape: {df.shape}")
    print(f"Initial columns: {df.columns.tolist()}")
    
    # Clean column names
    df = clean_df(df)
    
    # Standardize column names
    if 'Soil_Type' in df.columns:
        df.rename(columns={'Soil_Type': 'Soil Type'}, inplace=True)
    if 'Crop' in df.columns:
        df.rename(columns={'Crop': 'Crop Type'}, inplace=True)
    
    print(f"After cleaning: {df.columns.tolist()}")
    
    # Validate required columns exist
    required_features = ['Temperature', 'Humidity', 'Moisture', 'Soil_pH', 
                        'Rainfall', 'Soil Type', 'Crop Type']
    
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")
    
    # Strip whitespace from string columns
    df['Soil Type'] = df['Soil Type'].astype(str).str.strip()
    df['Crop Type'] = df['Crop Type'].astype(str).str.strip()
    
    # Check for data quality issues
    print(f"\nData quality check:")
    for col in required_features:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  WARNING: {col} has {null_count} null values")
            df[col].fillna(df[col].median() if df[col].dtype != 'object' else df[col].mode()[0], inplace=True)
    
    # CRITICAL: Analyze class distribution BEFORE encoding
    print(f"\n{'='*80}")
    print("CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    crop_counts = df['Crop Type'].value_counts()
    print(f"Total unique crops: {len(crop_counts)}")
    print(f"Total samples: {len(df)}")
    print(f"Samples per crop: min={crop_counts.min()}, max={crop_counts.max()}, mean={crop_counts.mean():.1f}")
    
    print(f"\nCrop distribution:")
    for crop, count in crop_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {crop:15} : {count:3} samples ({pct:5.2f}%)")
    
    # Check for severe imbalance
    max_count = crop_counts.max()
    min_count = crop_counts.min()
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 10:
        print(f"\n⚠️  CRITICAL WARNING: Severe class imbalance detected!")
        print(f"    Ratio: {imbalance_ratio:.1f}:1")
        print(f"    Most common: {crop_counts.idxmax()} ({max_count} samples)")
        print(f"    Least common: {crop_counts.idxmin()} ({min_count} samples)")
        print(f"    The model will likely be biased toward common crops.")
    
    # CRITICAL FIX: Data augmentation for small dataset
    if len(df) < 500:
        print(f"\n⚠️  Dataset is small ({len(df)} samples). Applying data augmentation...")
        df = augment_crop_data(df)
        print(f"    After augmentation: {len(df)} samples")
    
    # Encode categorical features
    encoders = {}
    for col in ['Soil Type', 'Crop Type']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"\nEncoded {col}: {len(le.classes_)} classes")
        print(f"  Classes: {le.classes_[:10]}{'...' if len(le.classes_) > 10 else ''}")
    
    # Define exact feature order
    feature_columns = [
        'Temperature', 
        'Humidity', 
        'Moisture', 
        'Soil_pH', 
        'Rainfall', 
        'Soil Type'
    ]
    
    X = df[feature_columns]
    y = df['Crop Type']
    
    print(f"\n{'='*80}")
    print("FINAL TRAINING DATA")
    print(f"{'='*80}")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Feature order: {X.columns.tolist()}")
    print(f"\nFeature statistics:")
    print(X.describe())
    
    return X, y, encoders


def augment_crop_data(df: pd.DataFrame, multiplier: int = 3) -> pd.DataFrame:
    """
    Augments crop data by adding slight variations to existing samples.
    This helps with small datasets to prevent overfitting to specific values.
    """
    print(f"  Augmenting data with {multiplier}x variations per sample...")
    
    augmented_dfs = [df.copy()]
    
    for i in range(multiplier - 1):
        df_aug = df.copy()
        
        # Add small random variations to numeric features
        for col in ['Temperature', 'Humidity', 'Moisture', 'Soil_pH', 'Rainfall']:
            # Add noise: ±5% of the value
            noise = df_aug[col] * np.random.uniform(-0.05, 0.05, size=len(df_aug))
            df_aug[col] = df_aug[col] + noise
            
            # Ensure values stay within reasonable bounds
            if col == 'Humidity' or col == 'Moisture':
                df_aug[col] = df_aug[col].clip(0, 100)
            elif col == 'Soil_pH':
                df_aug[col] = df_aug[col].clip(0, 14)
            elif col == 'Temperature':
                df_aug[col] = df_aug[col].clip(0, 50)
            elif col == 'Rainfall':
                df_aug[col] = df_aug[col].clip(0, 500)
        
        augmented_dfs.append(df_aug)
    
    result = pd.concat(augmented_dfs, ignore_index=True)
    print(f"  Original: {len(df)} samples -> Augmented: {len(result)} samples")
    
    return result


def train_crop_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains the Random Forest classifier for crop recommendation.
    
    CRITICAL FIXES:
    1. Proper train/test split for validation
    2. Model evaluation and diagnostics
    3. Optimized hyperparameters
    """
    print(f"\n{'='*80}")
    print("TRAINING CROP MODEL")
    print(f"{'='*80}")
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Check class distribution in training set
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nTraining class distribution:")
    print(f"  min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
    
    # CRITICAL: Optimized hyperparameters for small, imbalanced datasets
    model = RandomForestClassifier(
        n_estimators=300,            # More trees for stability
        max_depth=15,                # Limit depth to prevent overfitting
        min_samples_split=5,         # Require minimum samples to split
        min_samples_leaf=2,          # Require minimum samples in leaf
        max_features='sqrt',         # Use subset of features
        random_state=42,
        class_weight='balanced',     # CRITICAL: Handle imbalanced classes
        bootstrap=True,              # Use bootstrap sampling
        n_jobs=-1                    # Use all CPU cores
    )
    
    print(f"\nModel hyperparameters:")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  class_weight: {model.class_weight}")
    print(f"  max_features: {model.max_features}")
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # CRITICAL: Validate the model
    print(f"\n{'='*80}")
    print("MODEL VALIDATION")
    print(f"{'='*80}")
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    print(f"Testing accuracy:  {test_score:.4f} ({test_score*100:.2f}%)")
    
    if train_score - test_score > 0.2:
        print(f"\n⚠️  WARNING: Large gap between train and test accuracy!")
        print(f"    This indicates overfitting. Consider:")
        print(f"    - Collecting more data")
        print(f"    - Reducing max_depth")
        print(f"    - Increasing min_samples_split")
    
    # Feature importance analysis
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*80}")
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    for feat, imp in sorted_importance:
        bar = "█" * int(imp * 50)
        print(f"  {feat:15} {bar} {imp:.4f}")
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    
    # Check if model predicts only one class
    unique_predictions = np.unique(y_pred)
    print(f"\n{'='*80}")
    print("PREDICTION DIVERSITY CHECK")
    print(f"{'='*80}")
    print(f"Number of unique predictions: {len(unique_predictions)}/{len(np.unique(y))}")
    
    if len(unique_predictions) < len(np.unique(y)) * 0.5:
        print(f"\n⚠️  CRITICAL WARNING: Model predicts very few classes!")
        print(f"    Unique predictions: {len(unique_predictions)}")
        print(f"    Total classes: {len(np.unique(y))}")
        print(f"    The model is not learning to differentiate between crops properly.")
        
        # Show what it's predicting
        pred_counts = pd.Series(y_pred).value_counts()
        print(f"\n    Most common predictions:")
        for pred_label, count in pred_counts.head(5).items():
            pct = (count / len(y_pred)) * 100
            print(f"      Class {pred_label}: {count} times ({pct:.1f}%)")
    
    print(f"\n{'='*80}")
    print("MODEL TRAINING COMPLETE")
    print(f"{'='*80}\n")
    
    return model