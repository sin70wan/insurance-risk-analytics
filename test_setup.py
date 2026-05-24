"""
Test script to verify setup works
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

print("=" * 50)
print("Testing Insurance Risk Analytics Setup")
print("=" * 50)

# Test 1: Create sample data
print("\n[TEST 1] Creating sample data...")
np.random.seed(42)
n_samples = 1000

data = {
    'Province': np.random.choice(['Gauteng', 'Western Cape', 'KZN'], n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'TotalPremium': np.random.uniform(2000, 8000, n_samples),
    'TotalClaims': np.random.exponential(1000, n_samples),
    'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback'], n_samples),
}

df = pd.DataFrame(data)
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

print(f"✓ Created {len(df)} rows of sample data")

# Test 2: Basic analysis
print("\n[TEST 2] Running basic analysis...")
loss_ratio = df['TotalClaims'].sum() / df['TotalPremium'].sum()
print(f"✓ Overall Loss Ratio: {loss_ratio:.2%}")

# Test 3: Group by analysis
print("\n[TEST 3] Analysis by Province:")
province_stats = df.groupby('Province').agg({
    'LossRatio': 'mean',
    'Margin': 'mean'
}).round(3)
print(province_stats)

# Test 4: Create a simple plot
print("\n[TEST 4] Creating test plot...")
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='TotalClaims', by='Province', ax=ax)
ax.set_title('Claims Distribution by Province')
ax.set_ylabel('Total Claims')
plt.savefig('test_plot.png')
print("✓ Test plot saved as test_plot.png")

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED!")
print("=" * 50)
print("\nSystem is ready for Tasks 1 and 2!")