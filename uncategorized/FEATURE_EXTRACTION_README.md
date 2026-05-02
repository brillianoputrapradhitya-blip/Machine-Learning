# Satellite Feature Extraction for Crop Type and Phenophase Prediction

## Overview
This script extracts satellite imagery features from Sentinel-2 TIFF files and combines them with your labeled training data to create a feature matrix suitable for machine learning.

## What It Does

### Input Data
- **points_train_label.csv**: Training data with point locations (Longitude, Latitude) and labels (crop_type, phenophase_name)
- **ImageFiles/**: Directory containing Sentinel-2 satellite imagery organized by region and date

### Feature Extraction Process

1. **Spectral Band Values**: Extracts pixel values from all 12 Sentinel-2 bands:
   - B01: Coastal Aerosol (60m)
   - B02: Blue (10m)
   - B03: Green (10m)
   - B04: Red (10m)
   - B05-B07, B8A: Vegetation Red Edge (20m)
   - B08: Near-Infrared NIR (10m)
   - B09: Water Vapour (60m)
   - B11-B12: Short-Wave Infrared SWIR (20m)

2. **Spectral Indices**: Calculates agricultural indices from band combinations:
   - **NDVI** (Normalized Difference Vegetation Index): Vegetation health
   - **NDBI** (Normalized Difference Built-up Index): Built-up areas
   - **NDMI** (Normalized Difference Moisture Index): Soil moisture
   - **GNDVI** (Green Normalized Difference Vegetation Index): Chlorophyll content
   - **EVI** (Enhanced Vegetation Index): Improved vegetation measurement
   - **MNDWI** (Modified Normalized Difference Water Index): Water content

3. **Metadata**: Captures temporal proximity between point date and satellite image:
   - image_date: Date of closest satellite image
   - days_to_image: Number of days between point measurement and satellite image

### Output

**features_with_labels.csv** contains:
- **Identification columns**: point_id, Longitude, Latitude
- **Temporal columns**: phenophase_date, image_date, days_to_image, region
- **Spectral bands**: B_B01 through B_B12 (12 features)
- **Spectral indices**: NDVI, NDBI, NDMI, GNDVI, EVI, MNDWI (up to 6 features)
- **Labels**: crop_type, phenophase_name (for training)

**Total features**: ~27 features per sample

## Requirements

```bash
pip install rasterio pandas numpy
```

- **rasterio**: For reading GeoTIFF satellite imagery
- **pandas**: Data processing
- **numpy**: Numerical operations

## Usage

### Basic Usage
```python
python extract_features_from_tiff.py
```

The script will:
1. Load training data from `points_train_label.csv`
2. Index all satellite imagery in `ImageFiles/`
3. For each training point:
   - Find the closest satellite image date (within 60 days)
   - Extract spectral band values at the point location
   - Calculate spectral indices
4. Save results to `features_with_labels.csv`

### Configuration

Edit the configuration section in the script to change paths:

```python
INPUT_CSV = r'd:\!Reno\AIG\points_train_label.csv'
IMAGE_DIR = r'd:\!Reno\AIG\ImageFiles'
OUTPUT_CSV = r'd:\!Reno\AIG\features_with_labels.csv'
```

## Output Explanation

The output CSV contains one row per training sample with:

```
point_id | Longitude | Latitude | phenophase_date | image_date | days_to_image | region | crop_type | phenophase_name | B_B01 | B_B02 | ... | B_B12 | NDVI | NDBI | ... | MNDWI
```

### Example Row:
```
1 | 125.526 | 49.340 | 2018/6/7 | 2018-06-08 | 1 | region36 | soybean | Greenup | 458 | 1243 | ... | 892 | 0.65 | -0.12 | ... | 0.45
```

## Feature Statistics

- **Band values**: Unsigned 16-bit integers (0-10000 typical range)
- **Indices**: Float values between -1 and 1
- **Days to image**: 0-60 (filtered to maintain quality)
- **Missing values**: Very few if satellite image is available

## Next Steps for Model Training

1. **Data Preprocessing**:
   - Handle missing values (forward/backward fill or interpolation)
   - Normalize band values (divide by 10000)
   - Scale features using StandardScaler

2. **Feature Engineering**:
   - Create temporal features (day of year, month)
   - Create band ratios (B08/B04, etc.)
   - Apply PCA for dimensionality reduction

3. **Model Training**:
   - Use features for two-stage classification:
     - Stage 1: Predict `crop_type`
     - Stage 2: Predict `phenophase_name`
   - Try: Random Forest, XGBoost, Deep Learning

4. **Example Code**:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load features
df = pd.read_csv('features_with_labels.csv')

# Prepare features
feature_cols = [c for c in df.columns if c.startswith('B_') or c in ['NDVI', 'NDBI', 'NDMI', 'GNDVI', 'EVI', 'MNDWI']]
X = df[feature_cols]
y_crop = df['crop_type']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_scaled, y_crop)
```

## Troubleshooting

### "No close image date found"
- The satellite image may be too far from the point date (>60 days)
- Check that image dates are within the date range of your training data

### "Coordinate out of bounds"
- The point coordinates may not overlap with the satellite image area
- Verify longitude/latitude are in the correct coordinate system (WGS84)

### Missing band values
- Some bands may not be available for certain regions/dates
- Indices requiring missing bands won't be calculated (set to NaN)

## Feature Importance for Agriculture

For crop phenology prediction:
- **NDVI**: Most important - directly correlates with vegetation stage
- **NDMI**: Indicates water availability and stress
- **B08 (NIR)**: High correlation with biomass
- **B04 (Red)**: Complements NIR for vegetation detection
- **Temporal proximity** (days_to_image): Critical for accuracy

## References

- [Sentinel-2 Band Details](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument)
- [Spectral Indices Overview](https://www.indexdatabase.de/)
- [Remote Sensing for Agriculture](https://www.usgs.gov/faqs/what-remote-sensing-and-what-it-used)
