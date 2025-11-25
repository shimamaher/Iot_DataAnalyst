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
# 2. Data Cleaning - Modified for Asian Power Standards
# =============================================================================

print("\n" + "=" * 80)
print("Data Quality Check - Asian Power Grid Standards")
print("=" * 80)

# Display general information
print("\nGeneral data information:")
df.info()

# Check for missing values (but don't remove them yet)
print("\n\nMissing (Null) values in each column:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Number of Missing Values': missing_values,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Number of Missing Values'] > 0])

# Identify completely empty columns (but don't remove yet)
completely_empty = missing_df[missing_df['Percentage'] == 100.0].index.tolist()
if completely_empty:
    print(f"\nCompletely empty columns detected: {completely_empty}")
    print("Note: These columns will NOT be removed")

# Clean Country column - remove non-country values
non_countries = ['SolarPark', 'Substation', 'Site', 'Industrial', 'LoadHub', 'Center']
if 'Country' in df.columns:
    print(f"\n\nCleaning Country column...")
    print(f"Rows before cleaning: {len(df)}")
    df_cleaned = df[~df['Country'].isin(non_countries)].copy()
    print(f"Rows after removing non-country values: {len(df_cleaned)}")
    print(f"\nRemaining countries:")
    print(df_cleaned['Country'].value_counts())
else:
    df_cleaned = df.copy()

# Fill numeric missing values with median (instead of removing rows)
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
print("\n\nFilling missing numeric values with median:")
for col in numeric_columns:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        median_value = df_cleaned[col].median()
        df_cleaned[col].fillna(median_value, inplace=True)
        print(f"  {col}: Filled {missing_count} missing values with median ({median_value:.2f})")

# Remove duplicate rows
duplicates = df_cleaned.duplicated().sum()
print(f"\n\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")

# Define voltage standards for Asian continent (general range)
ASIA_VOLTAGE_STANDARD = {
    'min': 90,  # Minimum for Japan's 100V system
    'max': 250  # Maximum for 240V systems
}

print("\n\n" + "=" * 80)
print("OUTLIER DETECTION - Asian Continent Standards")
print("=" * 80)


# Detect voltage outliers based on Asian standards
def detect_voltage_outliers_asia(data, voltage_col='Voltage (V)'):
    """
    Detect voltage outliers based on Asian continent standards
    Acceptable range: 90V - 250V (covers all Asian voltage systems)
    """
    lower_bound = ASIA_VOLTAGE_STANDARD['min']
    upper_bound = ASIA_VOLTAGE_STANDARD['max']

    outliers = data[
        (data[voltage_col] < lower_bound) |
        (data[voltage_col] > upper_bound)
        ]

    print(f"\nAsian Voltage Standard Range: {lower_bound}V - {upper_bound}V")
    print(f"Total outliers detected: {len(outliers)} ({len(outliers) / len(data) * 100:.2f}%)")

    if len(outliers) > 0:
        print(f"Outlier voltage range: {outliers[voltage_col].min():.2f}V - {outliers[voltage_col].max():.2f}V")
        print("\nOutliers by country:")
        outlier_by_country = outliers.groupby('Country')[voltage_col].agg(['count', 'min', 'max'])
        print(outlier_by_country)

    return outliers


# Detect current outliers using IQR method
def detect_outliers_iqr(data, column):
    """
    Detect outliers using Interquartile Range (IQR) method
    Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


# Step 1: DETECT all outliers (don't remove yet)
print("\n1. VOLTAGE OUTLIERS:")
print("-" * 80)
if 'Voltage (V)' in df_cleaned.columns:
    voltage_outliers = detect_voltage_outliers_asia(df_cleaned)
else:
    voltage_outliers = pd.DataFrame()
    print("Voltage column not found")

print("\n2. CURRENT OUTLIERS:")
print("-" * 80)
if 'Current (A)' in df_cleaned.columns:
    current_outliers, current_lower, current_upper = detect_outliers_iqr(df_cleaned, 'Current (A)')
    print(f"Current (A) outliers: {len(current_outliers)} detected")
    print(f"Acceptable range: {current_lower:.2f}A - {current_upper:.2f}A")
    if len(current_outliers) > 0:
        print(
            f"Outlier range: {current_outliers['Current (A)'].min():.2f}A - {current_outliers['Current (A)'].max():.2f}A")
else:
    current_outliers = pd.DataFrame()
    print("Current column not found")

print("\n3. POWER CONSUMPTION OUTLIERS:")
print("-" * 80)
if 'Power Consumption' in df_cleaned.columns:
    power_outliers, power_lower, power_upper = detect_outliers_iqr(df_cleaned, 'Power Consumption')
    print(f"Power Consumption outliers: {len(power_outliers)} detected")
    print(f"Acceptable range: {power_lower:.2f} - {power_upper:.2f}")
    if len(power_outliers) > 0:
        print(
            f"Outlier range: {power_outliers['Power Consumption'].min():.2f} - {power_outliers['Power Consumption'].max():.2f}")
else:
    power_outliers = pd.DataFrame()
    print("Power Consumption column not found")

# Create a combined outlier index
all_outlier_indices = set()
if len(voltage_outliers) > 0:
    all_outlier_indices.update(voltage_outliers.index)
if len(current_outliers) > 0:
    all_outlier_indices.update(current_outliers.index)
if len(power_outliers) > 0:
    all_outlier_indices.update(power_outliers.index)

print("\n\n" + "=" * 80)
print("OUTLIER SUMMARY")
print("=" * 80)
print(f"Total unique rows with outliers: {len(all_outlier_indices)}")
print(f"Percentage of dataset: {len(all_outlier_indices) / len(df_cleaned) * 100:.2f}%")

# Display current dataset statistics before removal
print("\n\nCurrent dataset statistics by country:")
print("=" * 80)
if 'Country' in df_cleaned.columns and 'Voltage (V)' in df_cleaned.columns:
    voltage_stats = df_cleaned.groupby('Country')['Voltage (V)'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(voltage_stats)

print("\n\nCurrent dataset shape:")
print(f"Rows: {len(df_cleaned)}")
print(f"Columns: {len(df_cleaned.columns)}")

print("\n" + "=" * 80)
print("CLEANING PHASE 1 COMPLETED - No data removed yet")
print("=" * 80)
print("\nTo remove detected outliers, run the next code block")

# =============================================================================
# SEPARATE COMMAND: Remove outliers
# =============================================================================

print("\n\n" + "=" * 80)
print("REMOVING DETECTED OUTLIERS")
print("=" * 80)

# Store original size
original_size = len(df_cleaned)

# Remove all rows with outliers
df_final = df_cleaned.drop(index=list(all_outlier_indices))

removed_count = original_size - len(df_final)
print(f"\nRows removed: {removed_count}")
print(f"Rows remaining: {len(df_final)}")
print(f"Removal percentage: {removed_count / original_size * 100:.2f}%")

# Display final statistics
print("\n\nFinal dataset statistics by country:")
print("=" * 80)
if 'Country' in df_final.columns and 'Voltage (V)' in df_final.columns:
    final_voltage_stats = df_final.groupby('Country')['Voltage (V)'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(final_voltage_stats)

print("\n\nFinal dataset shape:")
print(f"Rows: {len(df_final)}")
print(f"Columns: {len(df_final.columns)}")

# Verify no outliers remain
print("\n\nVerifying cleaned data:")
print("-" * 80)
if 'Voltage (V)' in df_final.columns:
    voltage_check = len(df_final[
                            (df_final['Voltage (V)'] < ASIA_VOLTAGE_STANDARD['min']) |
                            (df_final['Voltage (V)'] > ASIA_VOLTAGE_STANDARD['max'])
                            ])
    print(f"Voltage outliers remaining: {voltage_check}")

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nCleaned dataset saved in variable: df_final")





