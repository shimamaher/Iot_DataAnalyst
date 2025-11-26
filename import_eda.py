
# =============================================================================
# STEP 1: Load Data
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_csv('C:\\0_DA\\Iot_DataAnalyst\\smart_grid_dataset_city_modified.csv') # Replace with your file name
print(" Data loaded successfully!")
print(f"Shape: {df.shape}")

# =============================================================================
# STEP 2: Initial Data Inspection
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: INITIAL DATA INSPECTION")
print("=" * 80)

# 2.1 Basic Information
print("\n--- 2.1 Dataset Information ---")
df.info()

# 2.2 First and Last Rows
print("\n--- 2.2 First 5 Rows ---")
print(df.head())

print("\n--- 2.3 Last 5 Rows ---")
print(df.tail())

# 2.3 Data Types
print("\n--- 2.4 Data Types ---")
print(df.dtypes.value_counts())

# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# 3.1 Statistical Summary
print("\n--- 3.1 Statistical Summary (Numeric Columns) ---")
print(df.describe())

print("\n--- 3.2 Statistical Summary (All Columns) ---")
print(df.describe(include='all'))

# 3.3 Missing Values Analysis
print("\n--- 3.3 Missing Values Analysis ---")
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': (df.isnull().sum() / len(df)) * 100,
    'Data_Type': df.dtypes
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
print(missing_df)

# 3.4 Duplicate Analysis
print("\n--- 3.4 Duplicate Rows ---")
duplicates = df.duplicated().sum()
print(f"Total duplicates: {duplicates} ({duplicates / len(df) * 100:.2f}%)")
if duplicates > 0:
    print("\nSample of duplicate rows:")
    print(df[df.duplicated(keep=False)].head(10))

# 3.5 Unique Values Analysis
print("\n--- 3.5 Unique Values in Each Column ---")
unique_df = pd.DataFrame({
    'Column': df.columns,
    'Unique_Values': [df[col].nunique() for col in df.columns],
    'Sample_Values': [df[col].unique()[:5] for col in df.columns]
})
print(unique_df)

# =============================================================================
# STEP 4: DISTRIBUTION ANALYSIS (Numeric Columns)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: DISTRIBUTION ANALYSIS")
print("=" * 80)

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    print(f"\n--- Distribution of {col} ---")
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Std: {df[col].std():.2f}")
    print(f"Min: {df[col].min():.2f}")
    print(f"Max: {df[col].max():.2f}")
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurtosis():.2f}")

# =============================================================================
# STEP 5: OUTLIER DETECTION (Before deciding to remove!)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: OUTLIER DETECTION ANALYSIS")
print("=" * 80)


def detect_outliers_iqr(data, column):
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers, lower, upper


outlier_summary = []
for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_summary.append({
        'Column': col,
        'Outlier_Count': len(outliers),
        'Outlier_Percent': len(outliers) / len(df) * 100,
        'Lower_Bound': lower,
        'Upper_Bound': upper,
        'Outlier_Min': outliers[col].min() if len(outliers) > 0 else None,
        'Outlier_Max': outliers[col].max() if len(outliers) > 0 else None
    })

outlier_df = pd.DataFrame(outlier_summary)
print("\n--- Outlier Summary ---")
print(outlier_df)

# =============================================================================
# STEP 6: VISUALIZATION (CRITICAL!)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: DATA VISUALIZATION")
print("=" * 80)

# 6.1 Distribution Plots
print("\n--- Creating distribution plots ---")
fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(15, 5 * len(numeric_cols)))
fig.suptitle('Distribution Analysis of Numeric Columns', fontsize=16, y=1.001)

for idx, col in enumerate(numeric_cols):
    # Histogram
    axes[idx, 0].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[idx, 0].set_title(f'Histogram: {col}')
    axes[idx, 0].set_xlabel(col)
    axes[idx, 0].set_ylabel('Frequency')
    axes[idx, 0].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
    axes[idx, 0].axvline(df[col].median(), color='green', linestyle='--', label='Median')
    axes[idx, 0].legend()

    # Boxplot
    axes[idx, 1].boxplot(df[col].dropna(), vert=False)
    axes[idx, 1].set_title(f'Boxplot: {col}')
    axes[idx, 1].set_xlabel(col)

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
print(" Saved: eda_distributions.png")
plt.show()

# 6.2 Correlation Heatmap
if len(numeric_cols) > 1:
    print("\n--- Creating correlation heatmap ---")
    plt.figure(figsize=(12, 10))
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Correlation Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('eda_correlation.png', dpi=300, bbox_inches='tight')
    print(" Saved: eda_correlation.png")
    plt.show()

# 6.3 Missing Values Heatmap
if df.isnull().sum().sum() > 0:
    print("\n--- Creating missing values heatmap ---")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig('eda_missing_values.png', dpi=300, bbox_inches='tight')
    print(" Saved: eda_missing_values.png")
    plt.show()

# 6.4 Categorical Analysis (if exists)
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\n--- Categorical Columns Analysis ---")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

        # Plot
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar', edgecolor='black')
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'eda_{col.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        print(f" Saved: eda_{col.lower().replace(' ', '_')}.png")
        plt.show()

# =============================================================================
# STEP 7: DECISION MAKING (Based on EDA findings)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: DATA CLEANING DECISIONS (Based on EDA)")
print("=" * 80)

print("""
Based on the EDA analysis above, you should now decide:

1. Missing Values:
    Are missing values random or systematic?
    Should we fill them (mean/median/mode) or drop rows?
    Is there a pattern in missing data?

2. Outliers:
    Are outliers real data or errors?
    Should we remove, cap, or keep them?
    Do they represent important extreme cases?

3. Duplicates:
    Are they true duplicates or legitimate repeated measurements?

4. Feature Selection:
    Which columns are relevant for analysis?
    Are there highly correlated features to remove?

5. Data Transformation:
    Do we need normalization/standardization?
    Do we need to handle skewed distributions?

 STOP HERE and review all visualizations and statistics!
 Make informed decisions based on domain knowledge!
""")

# =============================================================================
# STEP 8: GENERATE EDA REPORT
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: GENERATING EDA REPORT")
print("=" * 80)

report = f"""
EXPLORATORY DATA ANALYSIS REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATASET OVERVIEW
{'-' * 80}
   - Total Rows: {len(df):,}
   - Total Columns: {len(df.columns)}
   - Memory Usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB

2. DATA TYPES
{'-' * 80}
{df.dtypes.value_counts().to_string()}

3. MISSING VALUES
{'-' * 80}
   - Total Missing: {df.isnull().sum().sum():,}
   - Percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%

Columns with Missing Values:
{missing_df.to_string() if len(missing_df) > 0 else '   None'}

4. DUPLICATE ROWS
{'-' * 80}
   - Total Duplicates: {duplicates:,}
   - Percentage: {duplicates / len(df) * 100:.2f}%

5. OUTLIER ANALYSIS
{'-' * 80}
{outlier_df.to_string()}

6. NUMERIC COLUMNS SUMMARY
{'-' * 80}
{df[numeric_cols].describe().to_string()}

7. CATEGORICAL COLUMNS
{'-' * 80}
{chr(10).join([f'{col}: {df[col].nunique()} unique values' for col in categorical_cols]) if len(categorical_cols) > 0 else '   None'}

8. RECOMMENDATIONS
{'-' * 80}
   Based on this analysis, consider:

   a) Missing Values:
      - Review patterns in missing data
      - Decide on imputation strategy

   b) Outliers:
      - {outlier_df['Outlier_Count'].sum():,} total outliers detected
      - Review if they are valid data points

   c) Data Quality:
      - {duplicates} duplicate rows found
      - Check for data entry errors

   d) Next Steps:
      - Make informed cleaning decisions
      - Document all transformations
      - Validate cleaned data

{'=' * 80}
"""

with open('eda_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(" EDA Report saved: eda_report.txt")
print("\n" + "=" * 80)
print("EDA COMPLETED - Review all outputs before cleaning!")
print("=" * 80)


