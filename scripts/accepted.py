import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

# ========================= 1. LOAD DATA =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))

filename = "accepted_2007_to_2018Q4.csv.gz"
input_file = os.path.join(RAW_DIR, filename)
print(f"Loading file: {input_file}")

df = pd.read_csv(input_file, compression='gzip', low_memory=False)
print(f"Original shape: {df.shape}\n")

# ========================= 2. DROP COLUMNS WITH >50% NULLS =========================
NULL_THRESHOLD = 0.50
cols_before = df.shape[1]
cols_to_drop = [
    col for col in df.columns if df[col].isnull().sum() / len(df) > NULL_THRESHOLD
]
df.drop(columns=cols_to_drop, inplace=True)
print(
    f"Dropped {len(cols_to_drop)} columns (>50% nulls). Old shape: {cols_before}, New shape: {df.shape[1]}\n"
)

# ========================= 3. IMPUTE MISSING VALUES =========================
# Numeric columns → median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"Imputed numeric column '{col}' with median={med:.2f}")

# Object (text) columns → 'Unknown'
object_cols = df.select_dtypes(include=["object"]).columns
for col in object_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna("Unknown")
        print(f"Imputed text column '{col}' with 'Unknown'")
print()


# ========================= 4. FIX earliest_cr_line (multiple formats) =========================
def fix_earliest_cr_line(val):
    if pd.isna(val) or val == "Unknown":
        return np.nan
    val_str = str(val).strip()
    # Pattern: "Dec-99"
    m1 = re.match(r"^([A-Za-z]{3})-(\d{2})$", val_str)
    if m1:
        month, yr_short = m1.group(1), m1.group(2)
        yr_full = f"19{yr_short}" if int(yr_short) >= 90 else f"20{yr_short}"
        return f"{yr_full}-{month}-01"
    # Pattern: "03-Aug"
    m2 = re.match(r"^(\d{2})-([A-Za-z]{3})$", val_str)
    if m2:
        yr_short, month = m2.group(1), m2.group(2)
        yr_full = f"19{yr_short}" if int(yr_short) >= 90 else f"20{yr_short}"
        return f"{yr_full}-{month}-01"
    # Pattern: "Dec-1999"
    m3 = re.match(r"^([A-Za-z]{3})-(\d{4})$", val_str)
    if m3:
        month, yr_full = m3.group(1), m3.group(2)
        return f"{yr_full}-{month}-01"
    # Pattern: "2003-Aug"
    m4 = re.match(r"^(\d{4})-([A-Za-z]{3})$", val_str)
    if m4:
        yr_full, month = m4.group(1), m4.group(2)
        return f"{yr_full}-{month}-01"
    # Fallback
    try:
        return pd.to_datetime(val_str, errors="coerce").strftime("%Y-%b-01")
    except:
        return np.nan


if "earliest_cr_line" in df.columns:
    df["earliest_cr_line"] = df["earliest_cr_line"].apply(fix_earliest_cr_line)
    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"], format="%Y-%b-%d", errors="coerce"
    )
    # Fill any remaining NaT with median known date
    if df["earliest_cr_line"].isnull().any():
        known = df["earliest_cr_line"].dropna()
        if len(known) > 0:
            df["earliest_cr_line"] = df["earliest_cr_line"].fillna(known.median())
        else:
            df["earliest_cr_line"] = df["earliest_cr_line"].fillna(pd.Timestamp("2000-01-01"))
    print(
        f"'earliest_cr_line' fixed. Missing now: {df['earliest_cr_line'].isnull().sum()}\n"
    )


# ========================= 5. PARSE OTHER DATE COLUMNS (issue_d, last_pymnt_d, last_credit_pull_d) =========================
def robust_date_parser(series):
    series = series.astype(str).str.strip()
    formats = [
        "%b-%y",
        "%b-%Y",
        "%d-%b-%y",
        "%d-%b-%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed
    return pd.to_datetime(series, errors="coerce")


date_cols = ["issue_d", "last_pymnt_d", "last_credit_pull_d"]
for col in date_cols:
    if col in df.columns:
        print(f"Converting {col}...")
        df[col] = robust_date_parser(df[col])
        # Fill any remaining NaT with mode or median
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        print(f"  Missing after conversion: {df[col].isnull().sum()}")
print()

# ========================= 6. REMOVE DUPLICATE ROWS =========================
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicate rows: removed {before - len(df)}. New rows: {len(df)}")

# ========================= 7. OUTLIER REMOVAL (3*IQR) =========================
outlier_cols = [
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "dti",
    "revol_util",
    "fico_range_low",
    "fico_range_high",
]
outlier_mask = pd.Series(False, index=df.index)
for col in outlier_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_mask |= mask
        print(f"Outliers removed for {col}: {mask.sum()}")
before_out = len(df)
df = df[~outlier_mask]
print(f"Total outliers removed: {before_out - len(df)}. Remaining rows: {len(df)}\n")

# ========================= 8. CREATE BINNED VERSIONS (preserving original columns) =========================
bin_config = {
    "loan_amnt": (
        [0, 5000, 10000, 25000, np.inf],
        ["0-5K", "5K-10K", "10K-25K", "25K+"],
    ),
    "annual_inc": (
        [0, 30000, 60000, 100000, np.inf],
        ["<30K", "30K-60K", "60K-100K", "100K+"],
    ),
    "int_rate": ([0, 10, 15, 20, np.inf], ["0-10%", "10-15%", "15-20%", "20%+"]),
    "dti": ([0, 10, 20, 35, np.inf], ["0-10%", "10-20%", "20-35%", "35%+"]),
    "revol_util": ([0, 25, 50, 75, np.inf], ["0-25%", "25-50%", "50-75%", "75%+"]),
}
for col, (bins, labels) in bin_config.items():
    if col in df.columns:
        df[f"{col}_grp"] = pd.cut(df[col], bins=bins, labels=labels, right=True).astype(
            str
        )
        df[f"{col}_grp"] = df[f"{col}_grp"].replace("nan", "Unknown")
        print(f"Created binned column: {col}_grp")
print()

# ========================= 9. REMOVE DATA LEAKAGE COLUMNS =========================
leakage_cols = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_fico_range_high",
    "last_fico_range_low",
]
existing_leakage = [c for c in leakage_cols if c in df.columns]
if existing_leakage:
    df.drop(columns=existing_leakage, inplace=True)
    print(f"Dropped leakage columns: {existing_leakage}")
else:
    print("No leakage columns found.")

# ========================= 10. FIX FUTURE YEARS IN earliest_cr_line =========================
current_year = datetime.now().year
if "earliest_cr_line" in df.columns:
    df["earliest_cr_line"] = df["earliest_cr_line"].apply(
        lambda x: (
            x.replace(year=x.year - 100) if pd.notna(x) and x.year > current_year else x
        )
    )
    print("Fixed future years in 'earliest_cr_line'.")

# ========================= 11. ENSURE NO NULLS IN ANY _grp COLUMN =========================
grp_cols = [c for c in df.columns if c.endswith("_grp")]
for col in grp_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna("Unknown")
        print(f"Filled nulls in {col} with 'Unknown'")

# ========================= 12. SAVE FINAL CLEANED DATASET =========================
output_file = os.path.join(PROCESSED_DIR, "accepted_loans_final.csv")
df.to_csv(output_file, index=False)
print(f"\nFinal dataset saved as: {output_file}")

# ========================= 13. VALIDATION REPORT =========================
print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)
print("\nFirst 5 rows:")
print(df.head(5).to_string())
print("\nMissing values per column (should be 0):")
nulls = df.isnull().sum()
if nulls.sum() == 0:
    print("No missing values found.")
else:
    print(nulls[nulls > 0])
print(f"\nFinal shape: {df.shape}")
print("Accepted loans cleaning completed.")
