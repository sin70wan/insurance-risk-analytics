"""
Simplified EDA to test functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")

# Try to load real data, if not found create sample
try:
    df = pd.read_csv('data/insurance_data.csv')
    print(f"Loaded {len(df)} rows from data/insurance_data.csv")
except:
    print("Creating sample data...")
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'TransactionMonth': pd.date_range('2014-02-01', periods=n, freq='D'),
        'Province': np.random.choice(['Gauteng', 'Western Cape', 'KZN', 'Eastern Cape'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Bakkie'], n),
        'TotalPremium': np.random.uniform(2000, 8000, n),
        'TotalClaims': np.random.exponential(800, n),
        'Make': np.random.choice(['Toyota', 'VW', 'Ford', 'BMW'], n),
        'PostalCode': np.random.randint(1000, 9999, n),
    })
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    print(f"Created {len(df)} sample rows")

print("\n" + "="*60)
print("EDA SUMMARY")
print("="*60)

# Basic stats
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Key metrics
total_premium = df['TotalPremium'].sum()
total_claims = df['TotalClaims'].sum()
loss_ratio = total_claims / total_premium
margin = total_premium - total_claims

print(f"\n" + "="*40)
print("FINANCIAL METRICS")
print("="*40)
print(f"Total Premium: R {total_premium:,.2f}")
print(f"Total Claims:  R {total_claims:,.2f}")
print(f"Loss Ratio:    {loss_ratio:.2%}")
print(f"Margin:        R {margin:,.2f}")

# By Province
print(f"\n" + "="*40)
print("BY PROVINCE")
print("="*40)
province_stats = df.groupby('Province').agg({
    'TotalPremium': 'mean',
    'TotalClaims': 'mean',
    'LossRatio': 'mean',
    'Margin': 'mean'
}).round(2)
print(province_stats)

# By Gender
print(f"\n" + "="*40)
print("BY GENDER")
print("="*40)
gender_stats = df.groupby('Gender').agg({
    'TotalPremium': 'mean',
    'TotalClaims': 'mean',
    'LossRatio': 'mean'
}).round(2)
print(gender_stats)

# By Vehicle Type
print(f"\n" + "="*40)
print("BY VEHICLE TYPE")
print("="*40)
vehicle_stats = df.groupby('VehicleType').agg({
    'TotalPremium': 'mean',
    'TotalClaims': 'mean',
    'LossRatio': 'mean'
}).round(2)
print(vehicle_stats)

# Create visualizations
print(f"\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Loss Ratio by Province
province_loss = df.groupby('Province')['LossRatio'].mean().sort_values()
province_loss.plot(kind='barh', ax=axes[0,0], color='coral')
axes[0,0].set_title('Loss Ratio by Province')
axes[0,0].set_xlabel('Loss Ratio')

# Plot 2: Claims by Vehicle Type
df.boxplot(column='TotalClaims', by='VehicleType', ax=axes[0,1])
axes[0,1].set_title('Claims Distribution by Vehicle Type')
axes[0,1].set_ylabel('Total Claims')

# Plot 3: Premium vs Claims Scatter
axes[1,0].scatter(df['TotalPremium'], df['TotalClaims'], alpha=0.3)
axes[1,0].set_xlabel('Total Premium')
axes[1,0].set_ylabel('Total Claims')
axes[1,0].set_title('Premium vs Claims')

# Plot 4: Margin Distribution
df['Margin'].hist(ax=axes[1,1], bins=50, edgecolor='black')
axes[1,1].set_title('Margin Distribution')
axes[1,1].set_xlabel('Margin')

plt.tight_layout()
plt.savefig('eda_test_output.png', dpi=150, bbox_inches='tight')
print("✓ Saved eda_test_output.png")

print("\n" + "="*60)
print("✅ EDA COMPLETE!")
print("="*60)
print("\nKey Findings:")
print(f"1. Overall Loss Ratio: {loss_ratio:.2%}")
print(f"2. Best Province: {province_stats['LossRatio'].idxmin()} (Loss Ratio: {province_stats['LossRatio'].min():.2%})")
print(f"3. Worst Province: {province_stats['LossRatio'].idxmax()} (Loss Ratio: {province_stats['LossRatio'].max():.2%})")
print(f"4. Check 'eda_test_output.png' for visualizations")
