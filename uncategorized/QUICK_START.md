# Quick Start Guide: Satellite Feature Extraction

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in the AIG folder and run:

```bash
pip install rasterio pandas numpy scikit-learn
```

### Step 2: Extract Features
```bash
python extract_features_from_tiff.py
```

This will:
- Load your training data from `points_train_label.csv`
- Process all TIFF files in `ImageFiles/`
- Create `features_with_labels.csv` with extracted features

**Expected output:**
- ✓ Successfully extracted features for ~90-95% of samples
- ~27 features per sample (12 spectral bands + 6 indices + 9 metadata)
- CSV file ready for model training

### Step 3: Train Models
```bash
python example_model_training.py
```

This will:
- Split data into train/test sets
- Train two models:
  - Crop Type classifier
  - Phenophase Stage classifier
- Show accuracy metrics and feature importance

---

## 📊 What You Get

### Output CSV Structure

```
point_id | Longitude | Latitude | phenophase_date | image_date | region | 
crop_type | phenophase_name | 
B_B01 | B_B02 | ... | B_B12 |     (12 spectral bands)
NDVI | NDBI | NDMI | GNDVI | EVI | MNDWI |  (6 spectral indices)
days_to_image  (temporal proximity)
```

### Feature Meanings

**Spectral Bands (B_B01 - B_B12)**
- Pixel values from each Sentinel-2 band (0-10,000 range)
- Different wavelengths capture different vegetation properties

**Spectral Indices**
- **NDVI**: Vegetation greenness (0.0 = dead, 1.0 = healthy)
- **NDBI**: Built-up areas indicator
- **NDMI**: Moisture/water stress indicator
- **GNDVI**: Chlorophyll concentration
- **EVI**: Enhanced vegetation measurement
- **MNDWI**: Water/moisture content

**days_to_image**: How many days between your point measurement and the satellite pass

---

## 🎯 Model Performance Expectations

### For Crop Type (Soybean Detection)
- **Expected Accuracy**: 85-95%
- **Why good**: Clear spectral signature, seasonal patterns

### For Phenophase Prediction
- **Expected Accuracy**: 70-85%  
- **Why harder**: Subtle differences between stages, local variations

---

## 💡 Tips for Better Results

### 1. Data Preprocessing
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('features_with_labels.csv')

# Remove missing values
df = df.dropna()

# Normalize band values (important!)
band_cols = [c for c in df.columns if c.startswith('B_')]
df[band_cols] = df[band_cols] / 10000  # Sentinel-2 bands are 0-10000
```

### 2. Feature Engineering
```python
# Create additional features
df['NDVI_category'] = pd.cut(df['NDVI'], bins=5)  # Quantize NDVI
df['day_of_year'] = pd.to_datetime(df['phenophase_date']).dt.dayofyear
df['month'] = pd.to_datetime(df['phenophase_date']).dt.month

# Focus on key features
key_features = ['NDVI', 'B_B08', 'B_B04', 'B_B11', 'NDMI', 'days_to_image']
```

### 3. Model Selection
```python
# Try different algorithms
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# Start simple, then progress:
# 1. Random Forest (good baseline)
# 2. XGBoost (often better)
# 3. Neural Network (if you have enough data)
```

### 4. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 📁 Directory Structure

```
d:\!Reno\AIG\
├── points_train_label.csv              # Your training data (input)
├── ImageFiles/                          # Satellite images (input)
│   ├── region_train_1/
│   ├── region_train_2/
│   ├── region_train_3/
│   └── region_train_4/
├── extract_features_from_tiff.py       # Feature extraction script
├── example_model_training.py            # Model training example
├── setup_and_extract.py                 # Setup + extraction automation
├── FEATURE_EXTRACTION_README.md         # Detailed documentation
├── QUICK_START.md                       # This file
└── features_with_labels.csv             # OUTPUT: Your feature matrix
```

---

## 🔧 Troubleshooting

### "No module named 'rasterio'"
```bash
pip install --upgrade rasterio
```
Note: On Windows, you might need pre-compiled wheels:
```bash
pip install rasterio --prefer-binary
```

### "Coordinate out of bounds"
- Check that Longitude/Latitude are in WGS84 format
- Verify image bounds match your coordinates

### "No close image date found"
- Increase `days_diff` threshold in the script (default: 60 days)
- Check that image dates overlap with your point dates

### Low accuracy on phenophase
- Phenophase is harder to predict than crop type
- Try ensemble methods or collect more training data
- Use temporal features (day of year, month)

---

## 📚 Next Steps

1. **Review results**: Check `features_with_labels.csv`
2. **Tune parameters**: Adjust thresholds in extraction script
3. **Try advanced models**: Use XGBoost, LightGBM, or neural networks
4. **Deploy**: Save trained models and create inference pipeline

---

## 🎓 Learning Resources

- [Sentinel-2 on Wikipedia](https://en.wikipedia.org/wiki/Sentinel-2)
- [Spectral Index Database](https://www.indexdatabase.de/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Remote Sensing for Agriculture](https://www.gislounge.com/remote-sensing-agriculture/)

---

## ✅ Checklist

- [ ] Downloaded Sentinel-2 imagery to `ImageFiles/`
- [ ] Created `points_train_label.csv` with point locations and labels
- [ ] Installed required packages (`rasterio`, `pandas`, `numpy`)
- [ ] Ran `extract_features_from_tiff.py`
- [ ] Generated `features_with_labels.csv`
- [ ] Trained models with `example_model_training.py`
- [ ] Reviewed feature importance
- [ ] Ready to deploy!

---

## 📞 Support

If you encounter issues:
1. Check the detailed README: `FEATURE_EXTRACTION_README.md`
2. Review example script: `example_model_training.py`
3. Check error messages in console output
4. Verify file paths and data formats

Good luck! 🌾🛰️
