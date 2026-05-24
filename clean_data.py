"""
Clean the insurance dataset and create second DVC version
"""

import pandas as pd
import numpy as np

print("="*60)
print("CREATING CLEANED DATA VERSION (DVC v2)")
print("="*60)

# Load the raw data (pipe-separated as per the filename)
print("\nLoading raw data...")
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)

print(f"✓ Loaded {len(df):,} rows")
print(f"✓ Columns: {len(df.columns)}")

# Check for missing values
print("\n" + "-"*40)
print("MISSING VALUES ANALYSIS")
print("-"*40)
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
if len(missing_cols) > 0:
    print(f"Columns with missing values: {len(missing_cols)}")
    for col, count in missing_cols.head(10).items():
        print(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")
else:
    print("No missing values found!")

# Handle missing values
print("\n" + "-"*40)
print("HANDLING MISSING VALUES")
print("-"*40)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  {col}: filled with median")
        else:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
            print(f"  {col}: filled with mode")

# Remove outliers (cap at 99th percentile)
print("\n" + "-"*40)
print("OUTLIER TREATMENT")
print("-"*40)

if 'TotalClaims' in df.columns:
    q99 = df['TotalClaims'].quantile(0.99)
    before = df['TotalClaims'].max()
    df['TotalClaims'] = df['TotalClaims'].clip(upper=q99)
    print(f"  TotalClaims: capped from {before:,.2f} to {q99:,.2f}")

if 'TotalPremium' in df.columns:
    q99 = df['TotalPremium'].quantile(0.99)
    before = df['TotalPremium'].max()
    df['TotalPremium'] = df['TotalPremium'].clip(upper=q99)
    print(f"  TotalPremium: capped from {before:,.2f} to {q99:,.2f}")

# Add derived metrics
print("\n" + "-"*40)
print("ADDING DERIVED METRICS")
print("-"*40)

if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    print("  Added: LossRatio, Margin, HasClaim")

# Save cleaned version
print("\n" + "-"*40)
print("SAVING CLEANED DATA")
print("-"*40)

df.to_csv('data/insurance_data_cleaned.csv', index=False)
print(f"✓ Saved to data/insurance_data_cleaned.csv")

# Summary statistics
print("\n" + "="*60)
print("CLEANED DATA SUMMARY")
print("="*60)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Total Premium: R {df['TotalPremium'].sum():,.2f}")
print(f"Total Claims: R {df['TotalClaims'].sum():,.2f}")
print(f"Overall Loss Ratio: {(df['TotalClaims'].sum() / df['TotalPremium'].sum()):.2%}")
print(f"Claim Frequency: {(df['TotalClaims'] > 0).mean():.2%}")

if 'HasClaim' in df.columns:
    claims_df = df[df['HasClaim'] == 1]
    if len(claims_df) > 0:
        print(f"Average Severity: R {claims_df['TotalClaims'].mean():,.2f}")

print("\n" + "="*60)
print("✓ DVC v2 (cleaned data) created successfully!")
print("="*60)
