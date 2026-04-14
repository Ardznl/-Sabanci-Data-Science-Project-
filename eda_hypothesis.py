# =============================================================================
# DSA 210 - Phase 3: Data Cleaning, EDA & Hypothesis Testing
# Project: Evaluating the Impact of Startup Fundamentals and Investor Prestige
#          on Exit Success
# Author: Arad Zeinalifarid
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 5)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv('data/startup_valuation_dataset.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# =============================================================================
# 2. DATA CLEANING
# =============================================================================
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

df_clean = df.copy()

# Standardize column names (lowercase, replace spaces with underscores)
df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
print(f"Cleaned column names: {df_clean.columns.tolist()}")

# Identify the target column (exited / exit_status / similar)
# Try common naming conventions
possible_target = [c for c in df_clean.columns if 'exit' in c.lower()]
print(f"\nPossible target columns: {possible_target}")
target_col = possible_target[0] if possible_target else None

# Convert target to binary integer if it's boolean/string
if target_col:
    if df_clean[target_col].dtype == object:
        df_clean[target_col] = df_clean[target_col].map(
            {'True': 1, 'False': 0, 'true': 1, 'false': 0,
             'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0}
        )
    elif df_clean[target_col].dtype == bool:
        df_clean[target_col] = df_clean[target_col].astype(int)
    print(f"\nTarget variable '{target_col}' distribution:")
    print(df_clean[target_col].value_counts())
    print(f"Exit rate: {df_clean[target_col].mean()*100:.1f}%")

# Drop duplicates
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"\nDropped {before - len(df_clean)} duplicate rows.")

# Handle missing values
for col in df_clean.select_dtypes(include=[np.number]).columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

for col in df_clean.select_dtypes(include=['object']).columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna('Unknown', inplace=True)

print(f"\nMissing values after cleaning:\n{df_clean.isnull().sum().sum()} total")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Identify key numeric columns
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)

print(f"Numeric columns available: {num_cols}")

# Try to find revenue, funding, valuation, employee columns
def find_col(keywords, columns):
    for kw in keywords:
        matches = [c for c in columns if kw in c.lower()]
        if matches:
            return matches[0]
    return None

revenue_col    = find_col(['revenue'], df_clean.columns)
funding_col    = find_col(['funding_amount', 'amount_usd', 'funding_usd'], df_clean.columns)
valuation_col  = find_col(['valuation'], df_clean.columns)
employee_col   = find_col(['employee', 'headcount', 'staff'], df_clean.columns)
investor_col   = find_col(['lead_investor', 'investor'], df_clean.columns)
industry_col   = find_col(['industry', 'sector', 'vertical'], df_clean.columns)
region_col     = find_col(['region', 'country', 'location', 'geography'], df_clean.columns)
year_col       = find_col(['founding_year', 'founded', 'year'], df_clean.columns)

print(f"\nKey columns identified:")
print(f"  Revenue:    {revenue_col}")
print(f"  Funding:    {funding_col}")
print(f"  Valuation:  {valuation_col}")
print(f"  Employees:  {employee_col}")
print(f"  Investor:   {investor_col}")
print(f"  Industry:   {industry_col}")
print(f"  Region:     {region_col}")
print(f"  Year:       {year_col}")

# Revenue efficiency ratio
if revenue_col and employee_col:
    df_clean['revenue_per_employee'] = df_clean[revenue_col] / (df_clean[employee_col] + 1)
    print("\nCreated: revenue_per_employee")

# Valuation to funding multiple
if valuation_col and funding_col:
    df_clean['valuation_to_funding'] = df_clean[valuation_col] / (df_clean[funding_col] + 1)
    print("Created: valuation_to_funding")

# Investor prestige score: exit rate per investor
if investor_col and target_col:
    investor_exit_rate = df_clean.groupby(investor_col)[target_col].mean().rename('investor_prestige_score')
    df_clean = df_clean.merge(investor_exit_rate, on=investor_col, how='left')
    print("Created: investor_prestige_score")

    # Top tier = investors in top 25% by prestige
    prestige_threshold = df_clean['investor_prestige_score'].quantile(0.75)
    df_clean['is_top_tier_investor'] = (df_clean['investor_prestige_score'] >= prestige_threshold).astype(int)
    print("Created: is_top_tier_investor (binary)")

# Company age
if year_col:
    df_clean['company_age'] = 2024 - df_clean[year_col]
    print("Created: company_age")

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- 4.1 Target Variable Distribution ---
if target_col:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_clean[target_col].value_counts()
    bars = ax.bar(['Not Exited (0)', 'Exited (1)'],
                  counts.values,
                  color=['#5e81ac', '#a3be8c'], edgecolor='white', width=0.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{count:,}\n({count/len(df_clean)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11)
    ax.set_title('Distribution of Exit Status (Target Variable)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Startups')
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig('plot_01_target_distribution.png', bbox_inches='tight')
    plt.show()
    print("Saved: plot_01_target_distribution.png")

# --- 4.2 Numeric Feature Distributions ---
plot_cols = [c for c in [revenue_col, funding_col, valuation_col, employee_col] if c]
if plot_cols:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    colors = ['#5e81ac', '#81a1c1', '#88c0d0', '#8fbcbb']
    for i, col in enumerate(plot_cols[:4]):
        data = np.log1p(df_clean[col].dropna())
        axes[i].hist(data, bins=50, color=colors[i], edgecolor='white', alpha=0.85)
        axes[i].set_title(f'Distribution of log({col})', fontsize=11, fontweight='bold')
        axes[i].set_xlabel(f'log(1 + {col})')
        axes[i].set_ylabel('Count')
    for j in range(len(plot_cols), 4):
        axes[j].set_visible(False)
    plt.suptitle('Key Financial & Operational Metrics (Log Scale)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('plot_02_feature_distributions.png', bbox_inches='tight')
    plt.show()
    print("Saved: plot_02_feature_distributions.png")

# --- 4.3 Exit Rate by Industry ---
if industry_col and target_col:
    industry_exit = df_clean.groupby(industry_col)[target_col].agg(['mean', 'count']).reset_index()
    industry_exit.columns = [industry_col, 'exit_rate', 'count']
    industry_exit = industry_exit[industry_exit['count'] >= 50].sort_values('exit_rate', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(industry_exit)*0.5)))
    bars = ax.barh(industry_exit[industry_col], industry_exit['exit_rate'] * 100,
                   color='#5e81ac', edgecolor='white')
    ax.axvline(df_clean[target_col].mean() * 100, color='#bf616a', linestyle='--',
               linewidth=1.5, label=f"Overall avg ({df_clean[target_col].mean()*100:.1f}%)")
    for bar, rate in zip(bars, industry_exit['exit_rate']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{rate*100:.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Exit Rate (%)')
    ax.set_title('Exit Rate by Industry', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plot_03_exit_rate_by_industry.png', bbox_inches='tight')
    plt.show()
    print("Saved: plot_03_exit_rate_by_industry.png")

# --- 4.4 Exit Rate by Region ---
if region_col and target_col:
    region_exit = df_clean.groupby(region_col)[target_col].agg(['mean', 'count']).reset_index()
    region_exit.columns = [region_col, 'exit_rate', 'count']
    region_exit = region_exit[region_exit['count'] >= 50].sort_values('exit_rate', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(region_exit)*0.5)))
    ax.barh(region_exit[region_col], region_exit['exit_rate'] * 100, color='#81a1c1', edgecolor='white')
    ax.axvline(df_clean[target_col].mean() * 100, color='#bf616a', linestyle='--',
               linewidth=1.5, label=f"Overall avg ({df_clean[target_col].mean()*100:.1f}%)")
    ax.set_xlabel('Exit Rate (%)')
    ax.set_title('Exit Rate by Region', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plot_04_exit_rate_by_region.png', bbox_inches='tight')
    plt.show()
    print("Saved: plot_04_exit_rate_by_region.png")

# --- 4.5 Boxplots: Exited vs Not Exited ---
box_cols = [c for c in [revenue_col, funding_col, valuation_col] if c]
if box_cols and target_col:
    fig, axes = plt.subplots(1, len(box_cols), figsize=(5 * len(box_cols), 5))
    if len(box_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, box_cols):
        plot_df = df_clean[[target_col, col]].copy()
        plot_df[col] = np.log1p(plot_df[col])
        sns.boxplot(data=plot_df, x=target_col, y=col, palette=['#5e81ac', '#a3be8c'], ax=ax, width=0.5)
        ax.set_title(f'log({col}) by Exit Status', fontsize=11, fontweight='bold')
        ax.set_xlabel('Exited (0=No, 1=Yes)')
        ax.set_ylabel(f'log(1 + {col})')
    plt.suptitle('Financial Metrics: Exited vs Not Exited', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plot_05_boxplots_exited_vs_not.png', bbox_inches='tight')
    plt.show()
    print("Saved: plot_05_boxplots_exited_vs_not.png")

# --- 4.6 Correlation Heatmap ---
corr_cols = [c for c in [revenue_col, funding_col, valuation_col, employee_col,
                          'investor_prestige_score', 'revenue_per_employee',
                          'valuation_to_funding', 'company_age', target_col] if c and c in df_clean.columns]
if len(corr_cols) >= 3:
    corr_matrix = df_clean[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix of Key Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_06_correlation_heatmap.png', bbox_inches='tight')
    plt.show()
    print("Saved: plot_06_correlation_heatmap.png")

# --- 4.7 Investor Prestige vs Exit Rate ---
if 'investor_prestige_score' in df_clean.columns and target_col:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = pd.qcut(df_clean['investor_prestige_score'], q=5, duplicates='drop')
    prestige_exit = df_clean.groupby(bins)[target_col].mean() * 100
    prestige_exit.plot(kind='bar', color='#5e81ac', edgecolor='white', ax=ax)
    ax.set_title('Exit Rate by Investor Prestige Quintile', fontsize=13, fontweight='bold')
    ax.set_xlabel('Investor Prestige Score (Quintile)')
    ax.set_ylabel('Exit Rate (%)')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('plot_07_prestige_vs_exit_rate.png', bbox_inches='tight')
    plt.show()
    print("Saved: plot_07_prestige_vs_exit_rate.png")

# =============================================================================
# 5. HYPOTHESIS TESTING
# =============================================================================
print("\n" + "=" * 60)
print("HYPOTHESIS TESTING")
print("=" * 60)
alpha = 0.05

def print_result(h, stat_name, stat_val, p_val, alpha=0.05):
    print(f"\n--- {h} ---")
    print(f"  {stat_name} = {stat_val:.4f}")
    print(f"  p-value   = {p_val:.6f}")
    if p_val < alpha:
        print(f"  ✅ REJECT H0 (p < {alpha}): Statistically significant result.")
    else:
        print(f"  ❌ FAIL TO REJECT H0 (p >= {alpha}): Not statistically significant.")

# --- H1: Top-tier investors → higher exit rate? (Chi-Square) ---
if 'is_top_tier_investor' in df_clean.columns and target_col:
    contingency = pd.crosstab(df_clean['is_top_tier_investor'], df_clean[target_col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print_result(
        "H1: Top-tier investors have higher exit rates (Chi-Square)",
        "Chi2", chi2, p
    )
    top_exit    = df_clean[df_clean['is_top_tier_investor'] == 1][target_col].mean() * 100
    bottom_exit = df_clean[df_clean['is_top_tier_investor'] == 0][target_col].mean() * 100
    print(f"  Exit rate (top-tier investors):    {top_exit:.2f}%")
    print(f"  Exit rate (non-top-tier investors): {bottom_exit:.2f}%")

    # Visualize H1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Non-Top-Tier Investors', 'Top-Tier Investors'],
           [bottom_exit, top_exit],
           color=['#5e81ac', '#a3be8c'], edgecolor='white', width=0.5)
    ax.set_ylabel('Exit Rate (%)')
    ax.set_title(f'H1: Exit Rate by Investor Tier\n(Chi² = {chi2:.2f}, p = {p:.4f})',
                 fontsize=12, fontweight='bold')
    for i, v in enumerate([bottom_exit, top_exit]):
        ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('plot_08_H1_investor_tier.png', bbox_inches='tight')
    plt.show()

# --- H2: Exited startups have higher revenue? (Mann-Whitney U) ---
if revenue_col and target_col:
    exited     = df_clean[df_clean[target_col] == 1][revenue_col].dropna()
    not_exited = df_clean[df_clean[target_col] == 0][revenue_col].dropna()
    stat, p = stats.mannwhitneyu(exited, not_exited, alternative='greater')
    print_result(
        f"H2: Exited startups have higher {revenue_col} (Mann-Whitney U, one-tailed)",
        "U-stat", stat, p
    )
    print(f"  Median revenue (exited):     ${exited.median():,.0f}")
    print(f"  Median revenue (not exited): ${not_exited.median():,.0f}")

    # Visualize H2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Not Exited', 'Exited'],
           [not_exited.median(), exited.median()],
           color=['#5e81ac', '#a3be8c'], edgecolor='white', width=0.5)
    ax.set_ylabel(f'Median {revenue_col} (USD)')
    ax.set_title(f'H2: Median Revenue by Exit Status\n(Mann-Whitney U, p = {p:.4f})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_09_H2_revenue_exit.png', bbox_inches='tight')
    plt.show()

# --- H3: Exit rate differs across industries? (Chi-Square) ---
if industry_col and target_col:
    contingency2 = pd.crosstab(df_clean[industry_col], df_clean[target_col])
    contingency2 = contingency2[contingency2.sum(axis=1) >= 50]
    chi2_ind, p_ind, dof_ind, _ = stats.chi2_contingency(contingency2)
    print_result(
        "H3: Exit rate is different across industries (Chi-Square)",
        "Chi2", chi2_ind, p_ind
    )
    print(f"  Degrees of freedom: {dof_ind}")

    # Visualize H3
    industry_rates = (contingency2[1] / contingency2.sum(axis=1) * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, len(industry_rates)*0.5)))
    industry_rates.plot(kind='barh', color='#81a1c1', edgecolor='white', ax=ax)
    ax.axvline(df_clean[target_col].mean() * 100, color='#bf616a',
               linestyle='--', linewidth=1.5, label='Overall avg')
    ax.set_xlabel('Exit Rate (%)')
    ax.set_title(f'H3: Exit Rate by Industry\n(Chi² = {chi2_ind:.2f}, p = {p_ind:.4f})',
                 fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('plot_10_H3_industry_exit.png', bbox_inches='tight')
    plt.show()

# --- H4 (bonus): Exited startups have higher funding? (Mann-Whitney U) ---
if funding_col and target_col:
    exited_f     = df_clean[df_clean[target_col] == 1][funding_col].dropna()
    not_exited_f = df_clean[df_clean[target_col] == 0][funding_col].dropna()
    stat_f, p_f = stats.mannwhitneyu(exited_f, not_exited_f, alternative='greater')
    print_result(
        f"H4 (Bonus): Exited startups have higher {funding_col} (Mann-Whitney U)",
        "U-stat", stat_f, p_f
    )
    print(f"  Median funding (exited):     ${exited_f.median():,.0f}")
    print(f"  Median funding (not exited): ${not_exited_f.median():,.0f}")

# =============================================================================
# 6. SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3 COMPLETE - SUMMARY")
print("=" * 60)
print(f"  Dataset size after cleaning: {df_clean.shape}")
print(f"  Target variable: '{target_col}'")
print(f"  Exit rate: {df_clean[target_col].mean()*100:.1f}%")
print(f"  New engineered features: revenue_per_employee, valuation_to_funding,")
print(f"                           investor_prestige_score, is_top_tier_investor, company_age")
print(f"  Hypothesis tests run: H1 (Chi-Square), H2 (Mann-Whitney), H3 (Chi-Square), H4 (Mann-Whitney)")
print(f"\n  All plots saved as PNG files in the current directory.")
print(f"  ✅ Ready for GitHub commit.")
