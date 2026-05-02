import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.coords import BoundingBox
except ImportError:
    print("Warning: rasterio not installed. Install with: pip install rasterio")

# Configuration
INPUT_CSV = r'd:\!Reno\AIG\points_train_label.csv'
IMAGE_DIR = r'd:\!Reno\AIG\ImageFiles'
OUTPUT_CSV = r'd:\!Reno\AIG\features_with_labels.csv'

# Sentinel-2 band mapping
BANDS = {
    'B01': 'Coastal Aerosol',
    'B02': 'Blue',
    'B03': 'Green',
    'B04': 'Red',
    'B05': 'Vegetation Red Edge',
    'B06': 'Vegetation Red Edge',
    'B07': 'Vegetation Red Edge',
    'B08': 'NIR',
    'B8A': 'Vegetation Red Edge',
    'B09': 'Water Vapour',
    'B11': 'SWIR',
    'B12': 'SWIR',
}

def parse_tiff_filename(filename):
    """Extract region, image date, and Sentinel-2 band from a TIFF filename."""
    # Expected format:
    # region32_2018-09-18-00-00_2018-09-18-23-59_Sentinel-2_L2A_B09_(Raw).tiff
    #
    # A few files have inconsistent separators, e.g.
    # region17-2018-09-17-00_00_2018-09-17-23_59_..._B03_(Raw).tiff
    # so parse with regex instead of relying on fixed underscore positions.
    match = re.search(r'^(region\d+?)[_-]?(\d{4}-\d{2}-\d{2})', filename)
    band_match = re.search(r'_(B(?:\d{2}|8A))_', filename)

    if not match or not band_match:
        raise ValueError(f"Could not parse TIFF filename: {filename}")

    region = match.group(1)
    date_str = match.group(2)
    band = band_match.group(1)
    return region, date_str, band

def get_point_region_from_coordinates(lon, lat, image_bounds):
    """Match point coordinates to the region with overlapping bounds"""
    for region, bounds_info in image_bounds.items():
        if 'bounds' in bounds_info and bounds_info['bounds']:
            minx, miny, maxx, maxy = bounds_info['bounds']
            if minx <= lon <= maxx and miny <= lat <= maxy:
                return region
    return None

def get_image_bounds(tiff_path):
    """Get bounds of a TIFF image"""
    try:
        with rasterio.open(tiff_path) as src:
            bounds = src.bounds
            # Handle both projected and geographic coordinates
            left = bounds.left
            bottom = bounds.bottom
            right = bounds.right
            top = bounds.top
            
            # Validate bounds
            if left < right and bottom < top:
                return (left, bottom, right, top)
    except Exception as e:
        pass
    return None

def get_closest_date_images(point_date, region_images):
    """Find the closest date for a given region"""
    if not region_images:
        return None, float('inf')
    
    point_date_obj = datetime.strptime(point_date, '%Y/%m/%d')
    
    closest_date = None
    min_diff = float('inf')
    
    for img_date in region_images.keys():
        try:
            img_date_obj = datetime.strptime(img_date, '%Y-%m-%d')
            diff = abs((img_date_obj - point_date_obj).days)
            if diff < min_diff:
                min_diff = diff
                closest_date = img_date
        except:
            continue
    
    return closest_date, min_diff

def extract_pixel_value(raster_path, lon, lat):
    """Extract pixel value at given coordinates"""
    return extract_pixel_values(raster_path, [(lon, lat)])[0]

def extract_pixel_values(raster_path, coords):
    """Extract pixel values for many coordinates from one raster open."""
    values = [np.nan] * len(coords)

    try:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            valid_items = [
                (i, coord)
                for i, coord in enumerate(coords)
                if bounds.left <= coord[0] <= bounds.right
                and bounds.bottom <= coord[1] <= bounds.top
            ]

            if not valid_items:
                return values

            valid_indices = [i for i, _ in valid_items]
            valid_coords = [coord for _, coord in valid_items]

            # Read only the requested pixels. Grouping coordinates by raster keeps
            # the script from opening the same TIFF thousands of times.
            for output_idx, sampled in zip(
                valid_indices,
                src.sample(valid_coords, indexes=1, masked=True)
            ):
                value = sampled[0]

                if np.ma.is_masked(value):
                    continue

                value = float(value)
                if src.nodata is not None and np.isclose(value, src.nodata):
                    continue
                if not np.isfinite(value):
                    continue

                values[output_idx] = value
    except Exception as e:
        pass
    
    return values

def calculate_indices(bands_dict):
    """Calculate spectral indices from band values"""
    indices = {}
    
    try:
        # NDVI (Normalized Difference Vegetation Index) = (NIR - Red) / (NIR + Red)
        if 'B08' in bands_dict and 'B04' in bands_dict:
            b08 = bands_dict['B08']
            b04 = bands_dict['B04']
            if not np.isnan(b08) and not np.isnan(b04) and (b08 + b04) != 0:
                indices['NDVI'] = (b08 - b04) / (b08 + b04)
        
        # NDBI (Normalized Difference Built-up Index) = (SWIR - NIR) / (SWIR + NIR)
        if 'B11' in bands_dict and 'B08' in bands_dict:
            b11 = bands_dict['B11']
            b08 = bands_dict['B08']
            if not np.isnan(b11) and not np.isnan(b08) and (b11 + b08) != 0:
                indices['NDBI'] = (b11 - b08) / (b11 + b08)
        
        # NDMI (Normalized Difference Moisture Index) = (NIR - SWIR) / (NIR + SWIR)
        if 'B08' in bands_dict and 'B11' in bands_dict:
            b08 = bands_dict['B08']
            b11 = bands_dict['B11']
            if not np.isnan(b08) and not np.isnan(b11) and (b08 + b11) != 0:
                indices['NDMI'] = (b08 - b11) / (b08 + b11)
        
        # GNDVI (Green Normalized Difference Vegetation Index) = (NIR - Green) / (NIR + Green)
        if 'B08' in bands_dict and 'B03' in bands_dict:
            b08 = bands_dict['B08']
            b03 = bands_dict['B03']
            if not np.isnan(b08) and not np.isnan(b03) and (b08 + b03) != 0:
                indices['GNDVI'] = (b08 - b03) / (b08 + b03)
        
        # EVI (Enhanced Vegetation Index) = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        if 'B08' in bands_dict and 'B04' in bands_dict and 'B02' in bands_dict:
            b08 = bands_dict['B08']
            b04 = bands_dict['B04']
            b02 = bands_dict['B02']
            if not np.isnan(b08) and not np.isnan(b04) and not np.isnan(b02):
                denom = b08 + 6*b04 - 7.5*b02 + 1
                if denom != 0:
                    indices['EVI'] = 2.5 * (b08 - b04) / denom
        
        # MNDWI (Modified Normalized Difference Water Index) = (Green - SWIR) / (Green + SWIR)
        if 'B03' in bands_dict and 'B11' in bands_dict:
            b03 = bands_dict['B03']
            b11 = bands_dict['B11']
            if not np.isnan(b03) and not np.isnan(b11) and (b03 + b11) != 0:
                indices['MNDWI'] = (b03 - b11) / (b03 + b11)
    
    except Exception as e:
        print(f"Error calculating indices: {e}")
    
    return indices

def main():
    print("Loading training data...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} training samples")
    
    # Get list of image files organized by region and date
    print("\nIndexing satellite imagery...")
    image_structure = {}
    image_bounds = {}
    
    # Walk through ALL directories to find TIFF files
    for root, dirs, files in os.walk(IMAGE_DIR):
        for filename in files:
            if not filename.endswith('.tiff'):
                continue
            
            try:
                region, date_str, band = parse_tiff_filename(filename)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
            
            if region not in image_structure:
                image_structure[region] = {}
                image_bounds[region] = {'bounds': None}
            if date_str not in image_structure[region]:
                image_structure[region][date_str] = {}
            
            full_path = os.path.join(root, filename)
            image_structure[region][date_str][band] = full_path
            
            # Get bounds if not already set (from first available image for this region)
            if image_bounds[region]['bounds'] is None:
                bounds = get_image_bounds(full_path)
                if bounds:
                    image_bounds[region]['bounds'] = bounds
    
    print(f"Found {len(image_structure)} regions with satellite data")
    
    # Diagnostic: Show detailed region information
    print(f"\nRegion Details:")
    print(f"{'Region':<15} {'Dates':<8} {'Bands':<8} {'Bounds (Lon/Lat)':<50}")
    print("-" * 80)
    for region in sorted(image_structure.keys()):
        dates = list(image_structure[region].keys())
        if dates:
            sample_date = dates[0]
            bands = list(image_structure[region][sample_date].keys())
            bounds = image_bounds[region]['bounds']
            if bounds:
                bounds_str = f"[{bounds[0]:.2f}, {bounds[1]:.2f}] to [{bounds[2]:.2f}, {bounds[3]:.2f}]"
            else:
                bounds_str = "NOT READ"
            print(f"{region:<15} {len(dates):<8} {len(bands):<8} {bounds_str:<50}")
    
    # Check sample point against all regions
    print(f"\n{'='*80}")
    print("CHECKING SAMPLE POINT AGAINST REGIONS")
    print(f"{'='*80}")
    if len(df) > 0:
        sample_point = df.iloc[0]
        lon = sample_point['Longitude']
        lat = sample_point['Latitude']
        print(f"Sample point: Lon={lon:.4f}, Lat={lat:.4f}")
        print(f"Point date: {sample_point['phenophase_date']}")
        print(f"\nChecking against regions:")
        found_match = False
        for region in sorted(image_structure.keys()):
            bounds = image_bounds[region]['bounds']
            if bounds:
                minx, miny, maxx, maxy = bounds
                in_region = minx <= lon <= maxx and miny <= lat <= maxy
                status = "MATCH" if in_region else "out of bounds"
                print(f"  {region}: [{minx:.2f}, {miny:.2f}] to [{maxx:.2f}, {maxy:.2f}] ... {status}")
                if in_region:
                    found_match = True
        if not found_match:
            print(f"\nWARNING: Point does not match ANY region!")
            print(f"Point coordinates: ({lon}, {lat})")
            print(f"All region bounds above don't contain this point.")
    print()
    
    # Prepare feature rows and group points so each TIFF is opened once per group.
    features_list = []
    successful = 0
    failed = 0
    failed_reasons = {'no_region': 0, 'no_date': 0, 'few_bands': 0}

    candidate_rows = []
    grouped_rows = {}

    for idx, row in df.iterrows():
        point_id = row['point_id']
        lon = row['Longitude']
        lat = row['Latitude']
        point_date = row['phenophase_date']
        crop_type = row['crop_type']
        phenophase = row['phenophase_name']
        
        # Match point to region based on coordinates
        matched_region = get_point_region_from_coordinates(lon, lat, image_bounds)
        
        if matched_region is None:
            failed += 1
            failed_reasons['no_region'] += 1
            continue
        
        # Find closest date
        closest_date, days_diff = get_closest_date_images(point_date, image_structure[matched_region])
        
        if closest_date is None or days_diff > 60:  # Only use if within 60 days
            failed += 1
            failed_reasons['no_date'] += 1
            continue

        feature_row = {
            'point_id': point_id,
            'Longitude': lon,
            'Latitude': lat,
            'phenophase_date': point_date,
            'image_date': closest_date,
            'days_to_image': days_diff,
            'region': matched_region,
            'crop_type': crop_type,
            'phenophase_name': phenophase,
        }

        bands_dict = {}
        for band in sorted(BANDS.keys()):
            bands_dict[band] = np.nan
            feature_row[f'B_{band}'] = np.nan

        candidate = {
            'lon': lon,
            'lat': lat,
            'region': matched_region,
            'image_date': closest_date,
            'feature_row': feature_row,
            'bands_dict': bands_dict,
        }
        candidate_rows.append(candidate)
        grouped_rows.setdefault((matched_region, closest_date), []).append(candidate)

    print(
        f"Prepared {len(candidate_rows)} candidate rows in "
        f"{len(grouped_rows)} region/date groups"
    )

    for group_idx, ((region, image_date), rows) in enumerate(grouped_rows.items(), 1):
        if group_idx % 25 == 0 or group_idx == len(grouped_rows):
            print(
                f"Extracting group {group_idx}/{len(grouped_rows)} "
                f"({region}, {image_date}, {len(rows)} rows)"
            )

        coords = [(item['lon'], item['lat']) for item in rows]
        date_images = image_structure[region][image_date]

        for band in BANDS.keys():
            tiff_path = date_images.get(band)
            if not tiff_path:
                continue

            values = extract_pixel_values(tiff_path, coords)
            for item, value in zip(rows, values):
                item['bands_dict'][band] = value
                item['feature_row'][f'B_{band}'] = value

    for item in candidate_rows:
        bands_dict = item['bands_dict']

        # Skip if very few bands were successfully extracted
        valid_bands = sum(1 for v in bands_dict.values() if not np.isnan(v))
        if valid_bands < 5:
            failed += 1
            failed_reasons['few_bands'] += 1
            continue

        # Calculate spectral indices
        indices = calculate_indices(bands_dict)
        feature_row = item['feature_row']

        # Add spectral indices
        for idx_name, idx_value in indices.items():
            feature_row[idx_name] = idx_value
        
        features_list.append(feature_row)
        successful += 1
    
    # Create output dataframe
    if not features_list:
        print("\nERROR: No features were successfully extracted!")
        print("Possible causes:")
        print("  1. Check if TIFF files exist in ImageFiles/region_train_*/")
        print("  2. Verify point coordinates are within image bounds")
        print("  3. Ensure point dates are within 60 days of satellite image dates")
        print("  4. Check if rasterio can read the TIFF files")
        return None
    
    features_df = pd.DataFrame(features_list)
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully extracted: {successful}/{len(df)} points")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*successful/len(df):.1f}%")
    
    if failed > 0:
        print(f"\nFailure breakdown:")
        print(f"  No matching region: {failed_reasons['no_region']}")
        print(f"  Date too far (>60 days): {failed_reasons['no_date']}")
        print(f"  Too few valid bands (<5): {failed_reasons['few_bands']}")
    print(f"\nFeatures shape: {features_df.shape}")
    print(f"Columns: {len(features_df.columns)}")
    
    print(f"\nColumn names:")
    for i, col in enumerate(features_df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nFirst 3 rows of extracted features:")
    print(features_df.head(3).to_string())
    
    print(f"\nData statistics (band values):")
    band_cols = [c for c in features_df.columns if c.startswith('B_')]
    if band_cols:
        print(features_df[band_cols].describe())
    else:
        print("  No band columns found in extracted features")
    
    # Save to CSV
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved features to: {OUTPUT_CSV}")
    
    # Print missing values info
    print(f"\nMissing values per column:")
    missing = features_df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]}")
    
    return features_df

if __name__ == "__main__":
    features_df = main()
