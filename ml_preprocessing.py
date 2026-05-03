# =============================================================================
# DSA 210 - Phase 4: Machine Learning - Step 1: Preprocessing
# Project: Evaluating the Impact of Startup Fundamentals and Investor Prestige
#          on Exit Success
# Author: Arad Zeinalifarid
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 1: DATA PREPROCESSING & ENCODING")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv('data/startup_valuation_dataset.csv')
print(f"Loaded dataset: {df.shape}")

# =============================================================================
# 2. CLEAN & REPRODUCE FEATURES FROM PHASE 3
# =============================================================================
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

# Convert target to binary
df['exited'] = df['exited'].astype(int)

# Drop rows with missing exit_type only (45k nulls there are expected - not exited)
# Fill exit_type nulls with 'None'
df['exit_type'] = df['exit_type'].fillna('None')

# Engineer features (same as Phase 3)
df['revenue_per_employee'] = df['estimated_revenue_usd'] / (df['employee_count'] + 1)
df['valuation_to_funding']  = df['estimated_valuation_usd'] / (df['funding_amount_usd'] + 1)
df['company_age']            = 2024 - df['founded_year']

# Investor prestige score (exit rate per investor)
investor_exit_rate = df.groupby('lead_investor')['exited'].mean().rename('investor_prestige_score')
df = df.merge(investor_exit_rate, on='lead_investor', how='left')
prestige_threshold = df['investor_prestige_score'].quantile(0.75)
df['is_top_tier_investor'] = (df['investor_prestige_score'] >= prestige_threshold).astype(int)

print(f"Features engineered successfully.")

# =============================================================================
# 3. ENCODE CATEGORICAL VARIABLES
# =============================================================================
print("\nEncoding categorical variables...")

le_industry = LabelEncoder()
le_region   = LabelEncoder()
le_funding  = LabelEncoder()

df['industry_encoded']      = le_industry.fit_transform(df['industry'])
df['region_encoded']        = le_region.fit_transform(df['region'])
df['funding_round_encoded'] = le_funding.fit_transform(df['funding_round'])

print(f"  industry:      {df['industry'].nunique()} unique values encoded")
print(f"  region:        {df['region'].nunique()} unique values encoded")
print(f"  funding_round: {df['funding_round'].nunique()} unique values encoded")

# =============================================================================
# 4. SELECT FINAL FEATURE SET
# =============================================================================
FEATURES = [
    # Raw financials
    'funding_amount_usd',
    'estimated_revenue_usd',
    'estimated_valuation_usd',
    'employee_count',
    # Engineered fundamentals
    'revenue_per_employee',
    'valuation_to_funding',
    'company_age',
    # Investor prestige
    'investor_prestige_score',
    'is_top_tier_investor',
    # Encoded categoricals
    'industry_encoded',
    'region_encoded',
    'funding_round_encoded',
]

TARGET = 'exited'

X = df[FEATURES]
y = df[TARGET]

print(f"\nFeature set: {len(FEATURES)} features")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.1f}:1")

# =============================================================================
# 5. TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train exit rate: {y_train.mean()*100:.1f}% | Test exit rate: {y_test.mean()*100:.1f}%")

# =============================================================================
# 6. SCALE FEATURES
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"\nFeatures scaled with StandardScaler.")

# =============================================================================
# 7. SAVE PREPROCESSED DATA
# =============================================================================
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy',  X_test_scaled)
np.save('y_train.npy', y_train.values)
np.save('y_test.npy',  y_test.values)

# Save unscaled for tree models
X_train.to_csv('X_train_raw.csv', index=False)
X_test.to_csv('X_test_raw.csv',   index=False)
y_train.to_csv('y_train_raw.csv', index=False)
y_test.to_csv('y_test_raw.csv',   index=False)

# Save feature names and scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

print("\nSaved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("Saved: X_train_raw.csv, X_test_raw.csv (for tree models)")
print("Saved: scaler.pkl, feature_names.pkl")
print("\n✅ Preprocessing complete. Ready for model training.")
