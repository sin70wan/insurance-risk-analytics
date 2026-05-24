"""
EDA utilities for insurance data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def basic_stats(df: pd.DataFrame, numerical_cols: list):
    """
    Generate basic statistics for numerical columns.
    """
    return df[numerical_cols].describe()

def check_missing_values(df: pd.DataFrame):
    """
    Check and report missing values in the dataset.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    return missing_df[missing_df['Missing Count'] > 0]

def plot_distributions(df: pd.DataFrame, numerical_cols: list, save_path: str = None):
    """
    Plot histograms for numerical columns.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols[:6]):
        df[col].hist(ax=axes[i], bins=50, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, numerical_cols: list, save_path: str = None):
    """
    Plot correlation matrix for numerical columns.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_claims_by_province(df: pd.DataFrame, save_path: str = None):
    """
    Plot TotalClaims and LossRatio by Province.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Total claims by province
    claims_by_province = df.groupby('Province')['TotalClaims'].mean().sort_values()
    claims_by_province.plot(kind='barh', ax=axes[0], color='skyblue')
    axes[0].set_title('Average Total Claims by Province')
    axes[0].set_xlabel('Average Total Claims')
    
    # Loss ratio by province
    loss_by_province = df.groupby('Province')['LossRatio'].mean().sort_values()
    loss_by_province.plot(kind='barh', ax=axes[1], color='lightcoral')
    axes[1].set_title('Average Loss Ratio by Province')
    axes[1].set_xlabel('Loss Ratio')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def calculate_loss_ratio(df: pd.DataFrame) -> float:
    """
    Calculate overall portfolio loss ratio.
    """
    total_claims = df['TotalClaims'].sum()
    total_premium = df['TotalPremium'].sum()
    loss_ratio = total_claims / total_premium
    return loss_ratio

def calculate_claim_frequency(df: pd.DataFrame) -> float:
    """
    Calculate claim frequency (proportion of policies with claims).
    """
    return (df['TotalClaims'] > 0).mean()

def calculate_average_severity(df: pd.DataFrame) -> float:
    """
    Calculate average claim severity (average claim amount when claim occurred).
    """
    claims_df = df[df['TotalClaims'] > 0]
    if len(claims_df) > 0:
        return claims_df['TotalClaims'].mean()
    return 0.0

def outlier_detection_boxplot(df: pd.DataFrame, column: str, save_path: str = None):
    """
    Create boxplot for outlier detection.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df.boxplot(column=column, ax=ax)
    ax.set_title(f'Boxplot of {column} - Outlier Detection')
    ax.set_ylabel(column)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()