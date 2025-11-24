import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# تنظیمات نمایش
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# خواندن فایل CSV
df = pd.read_csv('C:\\0_DA\\Iot_DataAnalyst\\smart_grid_dataset_city_hourly_enriched.csv')  # نام فایل خود را جایگزین کنید

print("=" * 80)
print("داده‌ها با موفقیت بارگذاری شدند!")
print("=" * 80)
print(f"تعداد سطرها: {df.shape[0]}")
print(f"تعداد ستون‌ها: {df.shape[1]}")
print("\n5 سطر اول داده‌ها:")
print(df.head())
print("\nنام ستون‌ها:")
print(df.columns.tolist())

