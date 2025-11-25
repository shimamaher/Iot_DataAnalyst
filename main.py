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



# =============================================================================
# 2. Data Cleaning
# =============================================================================

print("\n" + "=" * 80)
print("Data Quality Check")
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

# Remove or fill missing values
# Method 1: Drop rows with missing important values
# Note: Check your actual column names first
important_columns = ['Voltage (V)', 'Current (A)']
# Add 'Power Consumption' only if it exists in your DataFrame
if 'Power Consumption' in df.columns:
    important_columns.append('Power Consumption')

df_cleaned = df.dropna(subset=important_columns)

# Method 2: Fill numeric values with median
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

# Check and remove duplicate rows
duplicates = df_cleaned.duplicated().sum()
print(f"\n\nNumber of duplicate rows: {duplicates}")
df_cleaned = df_cleaned.drop_duplicates()

# Detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\n\nChecking outliers in key columns:")
# Only check columns that exist in your DataFrame
columns_to_check = ['Voltage (V)', 'Current (A)']
if 'Power Consumption' in df_cleaned.columns:
    columns_to_check.append('Power Consumption')

for col in columns_to_check:
    if col in df_cleaned.columns:
        outliers, lower, upper = detect_outliers_iqr(df_cleaned, col)
        print(f"{col}: {len(outliers)} outliers (acceptable range: {lower:.2f} - {upper:.2f})")

print("\n" + "=" * 80)
print(f"Cleaning completed! Number of remaining rows: {len(df_cleaned)}")
print("=" * 80)




