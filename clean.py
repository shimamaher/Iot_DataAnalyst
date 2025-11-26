

import os
from datetime import datetime
import pandas as pd
import numpy as np

print("=" * 80)
print("ASIAN POWER GRID – CLEANING PIPELINE STARTED")
print("=" * 80)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

INPUT_FILE = "C:/0_DA/Iot_DataAnalyst/smart_grid_dataset_city_modified.csv"
OUTPUT_FOLDER = "cleaned_datasets"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"[INFO] Created folder: {OUTPUT_FOLDER}")
else:
    print(f"[INFO] Output folder exists: {OUTPUT_FOLDER}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# 2. LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 1] LOADING DATA")
print("=" * 80)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"[SUCCESS] Loaded file: {INPUT_FILE}")
    print(f"[INFO] Original shape: {df.shape}")
except FileNotFoundError:
    print(f"[ERROR] File not found: {INPUT_FILE}")
    raise
except Exception as e:
    print(f"[ERROR] Could not load file: {e}")
    raise

df_original = df.copy()

# =============================================================================
# 3. DATA QUALITY CHECK
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 2] DATA QUALITY CHECK")
print("=" * 80)

df.info()
print("\n[INFO] Missing values per column:")
print(df.isnull().sum())

# =============================================================================
# 4. HANDLE MISSING NUMERIC VALUES
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 3] HANDLING MISSING VALUES")
print("=" * 80)

df_cleaned = df.copy()
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)
        print(f"[FIX] {col}: Filled {missing_count} missing values with median {median_val}")

print(f"[DEBUG] Remaining missing values: {df_cleaned.isnull().sum().sum()}")

# =============================================================================
# 5. REMOVE DUPLICATES
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 4] DUPLICATE REMOVAL")
print("=" * 80)

duplicate_count = df_cleaned.duplicated().sum()
print(f"[INFO] Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"[SUCCESS] Removed {duplicate_count} duplicates")

# =============================================================================
# 6. OUTLIER DETECTION
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 5] OUTLIER DETECTION")
print("=" * 80)

ASIA_VOLTAGE_RANGE = {"min": 90, "max": 250}
outlier_indices = set()

# Voltage outliers (domain-based)
if "Voltage (V)" in df_cleaned.columns:
    v_outliers = df_cleaned[(df_cleaned["Voltage (V)"] < ASIA_VOLTAGE_RANGE["min"]) |
                            (df_cleaned["Voltage (V)"] > ASIA_VOLTAGE_RANGE["max"])]
    outlier_indices.update(v_outliers.index)
    print(f"[OUTLIER] Voltage: {len(v_outliers)}")

# IQR function
def detect_outliers_iqr(data, col):
    Q1, Q3 = data[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] < lower) | (data[col] > upper)]

# Current outliers
if "Current (A)" in df_cleaned.columns:
    c_outliers = detect_outliers_iqr(df_cleaned, "Current (A)")
    outlier_indices.update(c_outliers.index)
    print(f"[OUTLIER] Current (A): {len(c_outliers)}")

# Power consumption outliers
if "Power Consumption" in df_cleaned.columns:
    p_outliers = detect_outliers_iqr(df_cleaned, "Power Consumption")
    outlier_indices.update(p_outliers.index)
    print(f"[OUTLIER] Power Consumption: {len(p_outliers)}")

print(f"[SUMMARY] TOTAL OUTLIER ROWS: {len(outlier_indices)}")

# =============================================================================
# 7. REMOVE OUTLIERS
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 6] REMOVING OUTLIERS")
print("=" * 80)

before = len(df_cleaned)
df_final = df_cleaned.drop(index=outlier_indices)
after = len(df_final)

print(f"[INFO] Removed {before - after} rows")

# =============================================================================
# 8. CLEAN DecommissionStatus (Mixed Types)
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 7] CLEANING 'DecommissionStatus'")
print("=" * 80)

if "DecommissionStatus" in df_final.columns:

    # Create new categorical status
    df_final["OperationalStatus"] = df_final["DecommissionStatus"].apply(
        lambda x: "Operational" if x == "Operational" else "Decommissioned"
    )

    # Extract real dates
    df_final["DecommissionDate"] = pd.to_datetime(
        df_final["DecommissionStatus"], errors="coerce"
    )

    # Drop original column
    df_final.drop(columns=["DecommissionStatus"], inplace=True)

    print("[SUCCESS] Cleaned DecommissionStatus")

else:
    print("[WARN] Column 'DecommissionStatus' missing; skipping cleanup")

# =============================================================================
# 9. DATE CLEANING
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 8] DATE CLEANING")
print("=" * 80)

date_columns = ["Timestamp", "InstallationDate", "DecommissionDate"]

for col in date_columns:
    if col in df_final.columns:
        df_final[col] = pd.to_datetime(df_final[col], errors='coerce')
        print(f"[INFO] Converted {col} to datetime")

# =============================================================================
# 10. DROP USELESS COLUMNS (region)
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 9] DROPPING NON-INFORMATIVE COLUMNS")
print("=" * 80)

if "region" in df_final.columns:
    if df_final["region"].nunique() == 1:
        df_final.drop(columns=["region"], inplace=True)
        print("[INFO] Dropped column 'region' (only one category)")

# =============================================================================
# 11. CATEGORICAL ENCODING
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 10] CATEGORICAL ENCODING")
print("=" * 80)

categorical_cols = ["Name", "City", "Manufacturer", "Sensor_ID"]

for col in categorical_cols:
    if col in df_final.columns:
        df_final = pd.get_dummies(df_final, columns=[col], drop_first=True)
        print(f"[ENCODE] One-hot encoded {col}")

# Binary encoding for OperationalStatus
if "OperationalStatus" in df_final.columns:
    df_final["OperationalStatus"] = df_final["OperationalStatus"].map({
        "Operational": 1,
        "Decommissioned": 0
    })
    print("[ENCODE] Encoded OperationalStatus (Operational=1, Decommissioned=0)")

# =============================================================================
# 12. SAVE CLEANED DATA + REPORT
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 11] SAVING RESULTS")
print("=" * 80)

cleaned_file = os.path.join(OUTPUT_FOLDER, f"cleaned_data_{timestamp}.csv")
report_file = os.path.join(OUTPUT_FOLDER, f"cleaning_report_{timestamp}.txt")

df_final.to_csv(cleaned_file, index=False)
print(f"[SUCCESS] Cleaned data saved: {cleaned_file}")

report_text = f"""
CLEANING REPORT – ASIAN POWER GRID
===============================================
Timestamp: {timestamp}

Original rows:     {len(df_original)}
Final rows:        {len(df_final)}
Duplicates removed: {duplicate_count}
Outliers removed:   {before - after}

Processed columns:
- DecommissionStatus cleaned into OperationalStatus + DecommissionDate
- Dates converted to datetime
- region column dropped
- One-hot encoding for categorical columns
- Numeric missing values filled

Saved files:
- Cleaned CSV: {cleaned_file}
- Report:      {report_file}
"""

with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"[SUCCESS] Report saved: {report_file}")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 80)
print("CLEANING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
