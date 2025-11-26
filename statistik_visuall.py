import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings


# Load cleaned data
df_cleaned = pd.read_csv('cleaned_datasets/cleaned_data_20241126_143500.csv')
print(f"Loaded {len(df_cleaned):,} rows")



