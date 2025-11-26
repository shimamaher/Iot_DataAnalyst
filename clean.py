
# =============================================================================
# CLEANING PIPELINE – ASIAN POWER GRID
# Professional, Optimized, No Plots
# =============================================================================
# Only pandas + numpy + standard python libraries
# =============================================================================

import os
from datetime import datetime
import pandas as pd
import numpy as np
print(">>> RUNNING clean_save_new.py (CLEANING SCRIPT, NO PLOTS) <<<")
print("=" * 80)
print("ASIAN POWER GRID – CLEANING PIPELINE STARTED")
print("=" * 80)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Change this to your raw CSV file path
INPUT_FILE = "C:/0_DA/Iot_DataAnalyst/smart_grid_dataset_city_modified.csv"

# Folder where cleaned datasets and reports will be saved
OUTPUT_FOLDER = "cleaned_datasets"

# Create folder if not exists
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

print("\n[INFO] Dataset Information:")
df.info()

print("\n[INFO] Missing values per column:")
missing_values = df.isnull().sum()
print(missing_values)


# =============================================================================
# 4. FILL MISSING NUMERIC VALUES
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
        df_cleaned[col] = df_cleaned[col].fillna(median_val)
        print(f"[FIX] Filled {missing_count} missing values in '{col}' with median = {median_val}")

remaining_missing = df_cleaned.isnull().sum().sum()
print(f"[DEBUG] Remaining missing values: {remaining_missing}")


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
else:
    print("[INFO] No duplicates found")


# =============================================================================
# 6. OUTLIER DETECTION (Voltage + IQR)
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 5] OUTLIER DETECTION")
print("=" * 80)

ASIA_VOLTAGE_STANDARD = {"min": 90, "max": 250}
outlier_indices = set()


# ---- Voltage Standard Check ----
if "Voltage (V)" in df_cleaned.columns:
    voltage_outliers = df_cleaned[
        (df_cleaned["Voltage (V)"] < ASIA_VOLTAGE_STANDARD["min"]) |
        (df_cleaned["Voltage (V)"] > ASIA_VOLTAGE_STANDARD["max"])
    ]
    outlier_indices.update(voltage_outliers.index)
    print(f"[OUTLIER] Voltage outliers: {len(voltage_outliers)}")
else:
    print("[WARN] Column 'Voltage (V)' not found – skipping voltage check")


# ---- IQR Function ----
def detect_outliers_iqr(data, column):
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers, lower, upper


# ---- Current (A) IQR ----
if "Current (A)" in df_cleaned.columns:
    outliers, low, high = detect_outliers_iqr(df_cleaned, "Current (A)")
    outlier_indices.update(outliers.index)
    print(f"[OUTLIER] Current outliers: {len(outliers)}")
else:
    print("[WARN] Column 'Current (A)' not found")

# ---- Power Consumption IQR ----
if "Power Consumption" in df_cleaned.columns:
    outliers, low, high = detect_outliers_iqr(df_cleaned, "Power Consumption")
    outlier_indices.update(outliers.index)
    print(f"[OUTLIER] Power outliers: {len(outliers)}")
else:
    print("[WARN] Column 'Power Consumption' not found")


print(f"\n[SUMMARY] TOTAL outlier rows: {len(outlier_indices)}")


# =============================================================================
# 7. REMOVE OUTLIERS
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 6] REMOVING OUTLIERS")
print("=" * 80)

before = len(df_cleaned)
df_final = df_cleaned.drop(index=list(outlier_indices))
after = len(df_final)

print(f"[INFO] Before: {before} rows")
print(f"[INFO] After:  {after} rows")
print(f"[INFO] Removed: {before - after} rows")


# =============================================================================
# 8. SAVE CLEANED DATA + REPORT
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 7] SAVING RESULTS")
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
Missing handled:    Yes (median for numeric columns)

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
