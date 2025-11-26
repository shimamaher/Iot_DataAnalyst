import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Read CSV file
df = pd.read_csv('C:\\0_DA\\Iot_DataAnalyst\\smart_grid_dataset_city_modified.csv') # Replace with your file name

print("=" * 80)
print("Data loaded successfully!")
print("=" * 80)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nFirst 5 rows of data:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# =============================================================================
# 2. Data Cleaning - Optimized Version
# =============================================================================


print("\n" + "=" * 80)
print("Data Quality Check - Asian Power Grid Standards")
print("=" * 80)


# ===== Section 1: Initial Data Information =====
print("\nGeneral data information:")
df.info()


# ===== Section 2: Identify Missing Values =====
print("\n\nMissing (Null) values:")
missing_df = pd.DataFrame({
    'Missing': df.isnull().sum(),
    'Percentage': (df.isnull().sum() / len(df)) * 100
}).query('Missing > 0')
print(missing_df)


df_cleaned = df.copy()


# ===== Section 3: Fill Missing Values with Median =====
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
print("\n\nFilling missing numeric values with median:")
for col in numeric_cols:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        median_val = df_cleaned[col].median()
        df_cleaned[col].fillna(median_val, inplace=True)
        print(f"  {col}: Filled {missing_count} values with {median_val:.2f}")


# ===== Section 4: Remove Duplicate Rows =====
duplicates = df_cleaned.duplicated().sum()
if duplicates > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"\n\nRemoved {duplicates} duplicate rows")


# ===== Section 5: Define Asian Voltage Standards =====
ASIA_VOLTAGE_STANDARD = {'min': 90, 'max': 250}


print("\n\n" + "=" * 80)
print("OUTLIER DETECTION - Asian Standards")
print("=" * 80)


# ===== Section 6: Detect Outliers using IQR Method =====
def detect_outliers_iqr(data, column):
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers, lower, upper


all_outlier_indices = set()


# Voltage outliers
if 'Voltage (V)' in df_cleaned.columns:
    voltage_outliers = df_cleaned[
        (df_cleaned['Voltage (V)'] < ASIA_VOLTAGE_STANDARD['min']) |
        (df_cleaned['Voltage (V)'] > ASIA_VOLTAGE_STANDARD['max'])
    ]
    print(f"\nVoltage outliers: {len(voltage_outliers)} ({len(voltage_outliers)/len(df_cleaned)*100:.2f}%)")
    all_outlier_indices.update(voltage_outliers.index)


# Current outliers
if 'Current (A)' in df_cleaned.columns:
    current_outliers, c_lower, c_upper = detect_outliers_iqr(df_cleaned, 'Current (A)')
    print(f"Current outliers: {len(current_outliers)} (Range: {c_lower:.2f}A - {c_upper:.2f}A)")
    all_outlier_indices.update(current_outliers.index)


# Power outliers
if 'Power Consumption' in df_cleaned.columns:
    power_outliers, p_lower, p_upper = detect_outliers_iqr(df_cleaned, 'Power Consumption')
    print(f"Power outliers: {len(power_outliers)} (Range: {p_lower:.2f} - {p_upper:.2f})")
    all_outlier_indices.update(power_outliers.index)


print(f"\n\nTotal unique outlier rows: {len(all_outlier_indices)} ({len(all_outlier_indices)/len(df_cleaned)*100:.2f}%)")


# ===== Section 7: Remove Outliers =====
print("\n" + "=" * 80)
print("REMOVING OUTLIERS")
print("=" * 80)


original_size = len(df_cleaned)
df_final = df_cleaned.drop(index=list(all_outlier_indices))


print(f"\nRows removed: {original_size - len(df_final)}")
print(f"Rows remaining: {len(df_final)}")
print(f"Removal percentage: {(original_size - len(df_final))/original_size*100:.2f}%")


# ===== Section 8: Final Statistics =====
if 'Country' in df_final.columns and 'Voltage (V)' in df_final.columns:
    print("\n\nFinal statistics by country:")
    print(df_final.groupby('Country')['Voltage (V)'].agg(['count', 'mean', 'std', 'min', 'max']))


print(f"\n\nFinal shape: {df_final.shape}")
print("=" * 80)
print("DATA CLEANING COMPLETED")
print("=" * 80)

# =============================================================================
# Save Cleaned Data
# =============================================================================

import os
from datetime import datetime

print("\n" + "=" * 80)
print("SAVING CLEANED DATA")
print("=" * 80)

# Create output folder
output_folder = 'cleaned_datasets'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f" Created folder: {output_folder}")

# Generate filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = os.path.join(output_folder, f'cleaned_data_{timestamp}.csv')

# Save cleaned data
df_final.to_csv(output_file, index=False)

# Save statistics report
report_file = os.path.join(output_folder, f'cleaning_report_{timestamp}.txt')
report = f"""
Data Cleaning Report
{'=' * 80}

Original Data:
  - Rows: {len(df):,}
  - Columns: {len(df.columns)}

Cleaning Steps:
  - Duplicates removed: {df.duplicated().sum()}
  - Outliers removed: {len(df) - len(df_final):,}

Final Data:
  - Rows: {len(df_final):,}
  - Columns: {len(df_final.columns)}
  - Data loss: {(len(df) - len(df_final))/len(df)*100:.2f}%

File Information:
  - Output file: {output_file}
  - File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB
  - Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Column Names:
{', '.join(df_final.columns)}

{'=' * 80}
"""

with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n Files saved successfully:")
print(f"    Data: {output_file}")
print(f"    Report: {report_file}")
print(f"    Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
print("=" * 80)



