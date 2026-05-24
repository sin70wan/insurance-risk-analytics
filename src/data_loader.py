"""
Data loading utilities for insurance risk analytics
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_insurance_data(file_path: str = "data/insurance_data.csv") -> pd.DataFrame:
    """
    Load insurance dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with insurance data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        logger.warning(f"File {file_path} not found. Creating sample data...")
        return create_sample_data()

def create_sample_data() -> pd.DataFrame:
    """
    Create sample insurance data for initial development.
    
    Returns:
        Sample DataFrame with realistic insurance data
    """
    import numpy as np
    
    np.random.seed(42)
    n_samples = 10000
    
    provinces = ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape', 'Free State']
    genders = ['Male', 'Female']
    vehicle_types = ['Sedan', 'SUV', 'Hatchback', 'Bakkie']
    
    data = {
        'TransactionMonth': pd.date_range('2014-02-01', periods=n_samples, freq='D'),
        'Province': np.random.choice(provinces, n_samples),
        'Gender': np.random.choice(genders, n_samples),
        'VehicleType': np.random.choice(vehicle_types, n_samples),
        'TotalPremium': np.random.uniform(2000, 8000, n_samples),
        'TotalClaims': np.random.exponential(1000, n_samples),
        'CalculatedPremiumPerTerm': np.random.uniform(1800, 7500, n_samples),
        'CustomValueEstimate': np.random.uniform(50000, 500000, n_samples),
        'RegistrationYear': np.random.randint(1990, 2024, n_samples),
        'Make': np.random.choice(['Toyota', 'VW', 'Ford', 'BMW', 'Mercedes'], n_samples),
        'Model': np.random.choice(['Model S', 'Model X', 'Model 3'], n_samples),
        'PostalCode': np.random.randint(1000, 9999, n_samples),
        'SumInsured': np.random.uniform(50000, 1000000, n_samples),
        'CoverType': np.random.choice(['Comprehensive', 'Third Party'], n_samples),
        'IsVATRegistered': np.random.choice([True, False], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'AlarmImmobiliser': np.random.choice([True, False], n_samples),
        'TrackingDevice': np.random.choice([True, False], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add derived metrics
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/insurance_data.csv", index=False)
    logger.info(f"Created sample data with {len(df)} rows")
    
    return df

if __name__ == "__main__":
    df = load_insurance_data()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")