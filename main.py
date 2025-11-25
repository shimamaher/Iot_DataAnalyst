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
df = pd.read_csv('C:\\0_DA\\Iot_DataAnalyst\\smart_grid_dataset_city_hourly_enriched.csv') # Replace with your file name

print("=" * 80)
print("Data loaded successfully!")
print("=" * 80)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nFirst 5 rows of data:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())



# # =============================================================================
# # 2. Data Cleaning
# # =============================================================================
#
# print("\n" + "=" * 80)
# print("Data Quality Check")
# print("=" * 80)
#
# # Display general information
# print("\nGeneral data information:")
# df.info()
#
# # Check for missing values
# print("\n\nMissing (Null) values in each column:")
# missing_values = df.isnull().sum()
# missing_percent = (missing_values / len(df)) * 100
# missing_df = pd.DataFrame({
#     'Number of Missing Values': missing_values,
#     'Percentage': missing_percent
# })
# print(missing_df[missing_df['Number of Missing Values'] > 0])
#
# # Remove or fill missing values
# # Method 1: Drop rows with missing important values
# # Note: Check your actual column names first
# important_columns = ['Voltage (V)', 'Current (A)']
# # Add 'Power Consumption' only if it exists in your DataFrame
# if 'Power Consumption' in df.columns:
#     important_columns.append('Power Consumption')
#
# df_cleaned = df.dropna(subset=important_columns)
#
# # Method 2: Fill numeric values with median
# numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
# for col in numeric_columns:
#     if df_cleaned[col].isnull().sum() > 0:
#         df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
#
# # Check and remove duplicate rows
# duplicates = df_cleaned.duplicated().sum()
# print(f"\n\nNumber of duplicate rows: {duplicates}")
# df_cleaned = df_cleaned.drop_duplicates()
#
# # Detect outliers using IQR method
# def detect_outliers_iqr(data, column):
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
#     return outliers, lower_bound, upper_bound
#
# print("\n\nChecking outliers in key columns:")
# # Only check columns that exist in your DataFrame
# columns_to_check = ['Voltage (V)', 'Current (A)']
# if 'Power Consumption' in df_cleaned.columns:
#     columns_to_check.append('Power Consumption')
#
# for col in columns_to_check:
#     if col in df_cleaned.columns:
#         outliers, lower, upper = detect_outliers_iqr(df_cleaned, col)
#         print(f"{col}: {len(outliers)} outliers (acceptable range: {lower:.2f} - {upper:.2f})")
#
# print("\n" + "=" * 80)
# print(f"Cleaning completed! Number of remaining rows: {len(df_cleaned)}")
# print("=" * 80)

# Check unique countries in the dataset
print("Countries in the dataset:")
print("=" * 50)

if 'Country' in df.columns:
    country_counts = df['Country'].value_counts()
    print(f"\nTotal countries: {len(country_counts)}")
    print(f"\nCountry distribution:")
    print(country_counts)

    print(f"\n\nPercentage distribution:")
    country_percent = (country_counts / len(df)) * 100
    country_summary = pd.DataFrame({
        'Count': country_counts,
        'Percentage': country_percent
    })
    print(country_summary)

elif 'country' in df.columns:
    country_counts = df['country'].value_counts()
    print(f"\nTotal countries: {len(country_counts)}")
    print(f"\nCountry distribution:")
    print(country_counts)

    print(f"\n\nPercentage distribution:")
    country_percent = (country_counts / len(df)) * 100
    country_summary = pd.DataFrame({
        'Count': country_counts,
        'Percentage': country_percent
    })
    print(country_summary)

else:
    print("No 'Country' or 'country' column found in the dataset")
    print(f"\nAvailable columns:")
    print(df.columns.tolist())

# =============================================================================
# 2. Data Cleaning - Modified for Asian Power Standards
# =============================================================================

import numpy as np
import pandas as pd

print("\n" + "=" * 80)
print("Data Quality Check - Asian Power Grid Standards")
print("=" * 80)

# Display general information
print("\nGeneral data information:")
df.info()

# Check for missing values
print("\n\nMissing (Null) values in each column:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Number of Missing Values': missing_values,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Number of Missing Values'] > 0])

# Remove completely empty columns
completely_empty = missing_df[missing_df['Percentage'] == 100.0].index.tolist()
if completely_empty:
    print(f"\nRemoving completely empty columns: {completely_empty}")
    df = df.drop(columns=completely_empty)

# Clean Country column - remove non-country values
non_countries = ['SolarPark', 'Substation', 'Site', 'Industrial', 'LoadHub', 'Center']
if 'Country' in df.columns:
    print(f"\n\nCleaning Country column...")
    print(f"Rows before cleaning: {len(df)}")
    df = df[~df['Country'].isin(non_countries)]
    print(f"Rows after removing non-country values: {len(df)}")
    print(f"\nRemaining countries:")
    print(df['Country'].value_counts())

# Remove or fill missing values
important_columns = ['Voltage (V)', 'Current (A)']
if 'Power Consumption' in df.columns:
    important_columns.append('Power Consumption')

df_cleaned = df.dropna(subset=important_columns)

# Fill numeric missing values with median
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# Remove duplicate rows
duplicates = df_cleaned.duplicated().sum()
print(f"\n\nNumber of duplicate rows: {duplicates}")
df_cleaned = df_cleaned.drop_duplicates()

# Define voltage standards for each Asian country
VOLTAGE_STANDARDS = {
    'Japan': (90, 110),  # 100V system
    'Korea': (200, 240),  # 220V system
    'Thailand': (200, 240),  # 220V system
    'Vietnam': (200, 240),  # 220V system
    'Malaysia': (200, 250),  # 240V system
    'Singapur': (200, 250),  # 230V system
    'Philippinen': (200, 240),  # 220V system
    'Indonesien': (200, 240),  # 220V system
    'Indien': (200, 250),  # 230V system (but highly variable)
}


def detect_outliers_by_country(data, voltage_col='Voltage (V)', country_col='Country'):
    """
    Detect voltage outliers based on country-specific standards
    """
    all_outliers = []

    for country in data[country_col].unique():
        country_data = data[data[country_col] == country]

        if country in VOLTAGE_STANDARDS:
            lower_bound, upper_bound = VOLTAGE_STANDARDS[country]
        else:
            lower_bound, upper_bound = (200, 240)  # Default for 220V systems

        outliers = country_data[
            (country_data[voltage_col] < lower_bound) |
            (country_data[voltage_col] > upper_bound)
            ]

        all_outliers.append(outliers)

        print(f"\n{country}:")
        print(f"  Standard range: {lower_bound}V - {upper_bound}V")
        print(f"  Outliers: {len(outliers)} ({len(outliers) / len(country_data) * 100:.2f}%)")

        if len(outliers) > 0:
            print(f"  Min outlier: {outliers[voltage_col].min():.2f}V")
            print(f"  Max outlier: {outliers[voltage_col].max():.2f}V")

    return pd.concat(all_outliers) if all_outliers else pd.DataFrame()


print("\n\nChecking voltage outliers by country:")
print("=" * 80)

if 'Country' in df_cleaned.columns and 'Voltage (V)' in df_cleaned.columns:
    voltage_outliers = detect_outliers_by_country(df_cleaned)
    print(f"\n\nTotal voltage outliers across all countries: {len(voltage_outliers)}")


# Detect current outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


print("\n\nChecking current outliers:")
if 'Current (A)' in df_cleaned.columns:
    outliers, lower, upper = detect_outliers_iqr(df_cleaned, 'Current (A)')
    print(f"Current (A): {len(outliers)} outliers (acceptable range: {lower:.2f} - {upper:.2f})")

if 'Power Consumption' in df_cleaned.columns:
    outliers, lower, upper = detect_outliers_iqr(df_cleaned, 'Power Consumption')
    print(f"Power Consumption: {len(outliers)} outliers (acceptable range: {lower:.2f} - {upper:.2f})")

# Remove extreme voltage outliers (likely sensor errors)
print("\n\nRemoving extreme voltage outliers:")
extreme_low = 50  # Below 50V is sensor error
extreme_high = 400  # Above 400V is sensor error

before_count = len(df_cleaned)
df_cleaned = df_cleaned[
    (df_cleaned['Voltage (V)'] >= extreme_low) &
    (df_cleaned['Voltage (V)'] <= extreme_high)
    ]
removed = before_count - len(df_cleaned)
print(f"Removed {removed} rows with extreme voltage (< {extreme_low}V or > {extreme_high}V)")

print("\n" + "=" * 80)
print(f"Cleaning completed! Number of remaining rows: {len(df_cleaned)}")
print("=" * 80)

# Summary statistics by country
print("\n\nVoltage statistics by country:")
print("=" * 80)
voltage_stats = df_cleaned.groupby('Country')['Voltage (V)'].agg(['count', 'mean', 'std', 'min', 'max'])
print(voltage_stats)

print("\n\nFinal dataset shape:")
print(f"Rows: {len(df_cleaned)}")
print(f"Columns: {len(df_cleaned.columns)}")




