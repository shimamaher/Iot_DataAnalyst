# =============================================================================
# FULL ADVANCED CORRELATION ANALYSIS – FINAL VERSION (Excel-Friendly)
# =============================================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

plt.style.use("ggplot")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Build full path based on script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(
    BASE_DIR,
    "cleaned_datasets",
    "cleaned_data_20251126_194829.csv"
)

print("\n" + "=" * 80)
print("FULL ADVANCED CORRELATION ANALYSIS – FINAL VERSION")
print("=" * 80)

# =============================================================================
# 2. LOAD CLEANED DATA
# =============================================================================

df_cleaned = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"\nLoaded {len(df_cleaned):,} rows from: {INPUT_FILE}")

# Short alias
df = df_cleaned.copy()

# =============================================================================
# 3. CREATE 'HOUR' COLUMN FROM TIMESTAMP
# =============================================================================

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Hour"] = df["Timestamp"].dt.hour
    print("✓ Created 'Hour' column from Timestamp")
else:
    print(" Timestamp column not found – cannot create Hour column")

# =============================================================================
# 4. PAIRWISE CORRELATIONS (PEARSON – SPEARMAN – KENDALL)
# =============================================================================

print("\n" + "-" * 80)
print("1) Pairwise Correlations (Pearson, Spearman, Kendall)")
print("-" * 80)

# ELECTRICAL + ENERGY-WEATHER RELATIONSHIPS
correlation_pairs = [

    # Electrical relationships
    ("Voltage (V)", "Current (A)"),
    ("Current (A)", "Power Consumption (kW)"),
    ("Voltage (V)", "Power Consumption (kW)"),
    ("Power Factor", "Reactive Power (kVAR)"),

    # Energy–weather relationships requested by you
    ("Power Consumption (kW)", "Temperature (°C)"),
    ("Power Consumption (kW)", "Humidity (%)"),
    ("Power Consumption (kW)", "Wind Power (kW)"),
    ("Power Consumption (kW)", "Hour"),
]

methods = {
    "Pearson": pearsonr,
    "Spearman": spearmanr,
    "Kendall": kendalltau
}

correlation_results = []

for var1, var2 in correlation_pairs:
    if var1 in df.columns and var2 in df.columns:

        print(f"\nAnalyzing: {var1} ↔ {var2}")
        row = {"Variable 1": var1, "Variable 2": var2}

        s1 = df[var1].dropna()
        s2 = df[var2].dropna()
        idx = s1.index.intersection(s2.index)

        for name, func in methods.items():
            corr, p = func(s1.loc[idx], s2.loc[idx])
            print(f"  ➤ {name}: {corr:.4f} (p={p:.4e})")
            row[f"{name}_corr"] = corr
            row[f"{name}_pvalue"] = p

        correlation_results.append(row)

    else:
        print(f" Missing: {var1} or {var2}")

# SAVE RESULTS WITH SEMICOLON FOR EXCEL
pairwise_df = pd.DataFrame(correlation_results)
pairwise_df.to_csv(
    "final_full_correlation_results.csv",
    sep=";",                # IMPORTANT FOR GERMAN EXCEL
    index=False,
    encoding="utf-8-sig"
)

print("\n✓ Saved pairwise correlations to: final_full_correlation_results.csv (semicolon-separated)")


# =============================================================================
# 5. SPEARMAN CORRELATION MATRIX + HEATMAP
# =============================================================================

print("\n" + "-" * 80)
print("2) Spearman Correlation Matrix (All numeric features)")
print("-" * 80)

numeric_cols = df.select_dtypes(include=[np.number]).columns
spearman_corr = df[numeric_cols].corr(method="spearman")

plt.figure(figsize=(16, 12))
sns.heatmap(
    spearman_corr,
    annot=False,
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Spearman Correlation Matrix – Numeric Features", fontsize=16)
plt.tight_layout()
plt.savefig("7_spearman_correlation.png", dpi=300, bbox_inches="tight")
plt.show()

print("✓ Saved Spearman heatmap → 7_spearman_correlation.png")


# =============================================================================
# 6. PARTIAL CORRELATION (Voltage – Current | controlling Power Consumption)
# =============================================================================

print("\n" + "-" * 80)
print("3) Partial Correlation")
print("-" * 80)

def partial_correlation(df, x, y, z):
    """Calculate partial correlation between x and y controlling for z."""
    data = df[[x, y, z]].dropna()

    r_xy, _ = pearsonr(data[x], data[y])
    r_xz, _ = pearsonr(data[x], data[z])
    r_yz, _ = pearsonr(data[y], data[z])

    num = r_xy - r_xz * r_yz
    den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    return num / den if den != 0 else np.nan

if all(col in df.columns for col in ["Voltage (V)", "Current (A)", "Power Consumption (kW)"]):
    pcorr = partial_correlation(df, "Voltage (V)", "Current (A)", "Power Consumption (kW)")
    print(f"Partial correlation (Voltage–Current | Power Consumption): {pcorr:.4f}")
else:
    print(" Required columns for partial correlation not found!")


# =============================================================================
# 7. ROLLING CORRELATION (Voltage vs Current over Time)
# =============================================================================

print("\n" + "-" * 80)
print("4) Rolling Correlation – Voltage vs Current")
print("-" * 80)

if "Voltage (V)" in df.columns and "Current (A)" in df.columns:

    window_size = min(50, max(10, len(df) // 10))

    rolling_corr = df["Voltage (V)"].rolling(window=window_size)\
                                   .corr(df["Current (A)"])

    plt.figure(figsize=(14, 6))
    plt.plot(rolling_corr, linewidth=2)
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    plt.title(f"Rolling Correlation: Voltage vs Current (Window={window_size})", fontsize=14)
    plt.xlabel("Index")
    plt.ylabel("Correlation Coefficient")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("8_rolling_correlation.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Saved rolling correlation → 8_rolling_correlation.png")

else:
    print(" Voltage or Current column missing!")


print("\n" + "=" * 80)
print("ADVANCED CORRELATION ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
