"""
Example: Training Models with Extracted Satellite Features

This script demonstrates how to use the extracted satellite features
to train crop type and phenophase prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_CSV = r'd:\!Reno\AIG\features_with_labels.csv'

def load_and_prepare_data():
    """Load extracted features and prepare for training"""
    print("Loading feature data...")
    df = pd.read_csv(FEATURES_CSV)
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    print(f"\nDataset shape: {df.shape}")
    print(f"Unique crop types: {df['crop_type'].nunique()}")
    print(f"Unique phenophases: {df['phenophase_name'].nunique()}")
    
    # Define feature columns (spectral bands and indices)
    spectral_features = [c for c in df.columns if c.startswith('B_')]
    index_features = ['NDVI', 'NDBI', 'NDMI', 'GNDVI', 'EVI', 'MNDWI']
    temporal_features = ['days_to_image']
    
    feature_cols = spectral_features + [f for f in index_features if f in df.columns] + temporal_features
    
    print(f"\nFeatures to use:")
    print(f"  Spectral bands: {len(spectral_features)}")
    print(f"  Spectral indices: {len([f for f in index_features if f in df.columns])}")
    print(f"  Temporal features: {len(temporal_features)}")
    print(f"  Total: {len(feature_cols)} features")
    
    # Handle missing values
    print(f"\nHandling missing values...")
    initial_missing = df[feature_cols].isnull().sum().sum()
    print(f"  Missing values: {initial_missing}")
    
    # Drop rows with missing features
    df_clean = df.dropna(subset=feature_cols)
    
    print(f"  Rows after removing missing: {len(df_clean)} (removed {len(df) - len(df_clean)})")
    
    X = df_clean[feature_cols]
    y_crop = df_clean['crop_type']
    y_pheno = df_clean['phenophase_name']
    
    return X, y_crop, y_pheno, df_clean, feature_cols

def normalize_features(X_train, X_test):
    """Normalize features using StandardScaler"""
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_crop_type_model(X_train, X_test, y_train, y_test):
    """Train Random Forest model for crop type prediction"""
    print("\n" + "=" * 70)
    print("TRAINING CROP TYPE PREDICTION MODEL")
    print("=" * 70)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model trained successfully")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns if hasattr(X_train, 'columns') else [f"Feature_{i}" for i in range(X_train.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, accuracy, feature_importance

def train_phenophase_model(X_train, X_test, y_train, y_test):
    """Train Random Forest model for phenophase prediction"""
    print("\n" + "=" * 70)
    print("TRAINING PHENOPHASE PREDICTION MODEL")
    print("=" * 70)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model trained successfully")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns if hasattr(X_train, 'columns') else [f"Feature_{i}" for i in range(X_train.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, accuracy, feature_importance

def main():
    print("=" * 70)
    print("SATELLITE IMAGERY FEATURE-BASED MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    X, y_crop, y_pheno, df_clean, feature_cols = load_and_prepare_data()
    X = pd.DataFrame(X, columns=feature_cols)
    
    # Display statistics
    print(f"\nFeature Statistics:")
    print(X.describe())
    
    # Split data
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    
    X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
        X, y_crop, test_size=0.2, random_state=42, stratify=y_crop
    )
    
    X_train_pheno, X_test_pheno, y_train_pheno, y_test_pheno = train_test_split(
        X, y_pheno, test_size=0.2, random_state=42, stratify=y_pheno
    )
    
    print(f"\nTrain-test split (80-20):")
    print(f"  Training: {len(X_train_crop)} samples")
    print(f"  Testing: {len(X_test_crop)} samples")
    
    # Normalize
    X_train_crop_scaled, X_test_crop_scaled, scaler_crop = normalize_features(X_train_crop, X_test_crop)
    X_train_pheno_scaled, X_test_pheno_scaled, scaler_pheno = normalize_features(X_train_pheno, X_test_pheno)
    
    # Train models
    crop_model, crop_acc, crop_importance = train_crop_type_model(
        X_train_crop_scaled, X_test_crop_scaled, y_train_crop, y_test_crop
    )
    
    pheno_model, pheno_acc, pheno_importance = train_phenophase_model(
        X_train_pheno_scaled, X_test_pheno_scaled, y_train_pheno, y_test_pheno
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nCrop Type Prediction:")
    print(f"  Accuracy: {crop_acc:.4f}")
    print(f"  Samples: {len(X_train_crop)} training, {len(X_test_crop)} testing")
    
    print(f"\nPhenophase Prediction:")
    print(f"  Accuracy: {pheno_acc:.4f}")
    print(f"  Samples: {len(X_train_pheno)} training, {len(X_test_pheno)} testing")
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"  - Spectral bands (B02-B12): 12 features")
    print(f"  - Spectral indices: up to 6 features")
    print(f"  - Temporal: 1 feature (days_to_image)")
    
    print("\n✓ Training complete!")
    print("\nNext steps:")
    print("1. Save models for deployment")
    print("2. Fine-tune hyperparameters")
    print("3. Try other algorithms (XGBoost, Deep Learning)")
    print("4. Analyze feature importance for insights")

if __name__ == "__main__":
    main()
