#!/usr/bin/env python
"""
Setup script to install dependencies and run feature extraction
"""
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    packages = [
        'rasterio',
        'pandas',
        'numpy',
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def main():
    print("=" * 70)
    print("SATELLITE IMAGERY FEATURE EXTRACTION SETUP")
    print("=" * 70)
    
    # Install dependencies
    install_requirements()
    
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES FROM SATELLITE IMAGERY")
    print("=" * 70 + "\n")
    
    # Run feature extraction
    try:
        from extract_features_from_tiff import main as extract_main
        extract_main()
        print("\n" + "=" * 70)
        print("✓ FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nOutput file: features_with_labels.csv")
        print("\nNext steps:")
        print("1. Review FEATURE_EXTRACTION_README.md for details")
        print("2. Load the CSV in your notebook for model training")
        print("3. Use the extracted features with your ML pipeline")
    except Exception as e:
        print(f"\n✗ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
