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


