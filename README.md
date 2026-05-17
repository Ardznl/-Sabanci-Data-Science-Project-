# Evaluating the Impact of Startup Fundamentals and Investor Prestige on Exit Success

## Project Overview
This repository contains the course project for **Sabanci University - DSA 210: Introduction to Data Science**.

The goal of this research is to investigate whether a startup's path to a liquidity event (IPO or acquisition) is driven more by its **internal business fundamentals** (revenue, growth, efficiency) or the **prestige of its lead investors** (network effects). Using machine learning, we aim to decouple these drivers and identify the most reliable predictors of startup success.

---

## Dataset
The project utilizes the **Startup_Valuation_Dataset (2010–2024)** sourced from Kaggle.
- **Samples:** 50,000 unique startup records.
- **Scope:** Global funding activity across 7 key industries (AI/ML, Fintech, SaaS, etc.).
- **Key Features:** Revenue, employee count, valuation, lead investor, funding round details.
- **Target Variable:** `exited` (Binary classification — 14.7% exit rate).

---

## Current Status
- [x] **Phase 1:** Project Proposal Submitted.
- [x] **Phase 2:** Data Cleaning & Exploratory Data Analysis (EDA).
- [x] **Phase 3:** Feature Engineering, EDA & Hypothesis Testing.
- [x] **Phase 4:** Machine Learning Models & Interpretability Analysis.
- [x] **Phase 5:** Final Report.

---

## Repository Structure
```
├── data/
│   └── startup_valuation_dataset.csv      # Raw dataset (Kaggle)
├── eda_hypothesis.py                       # Phase 3: EDA & hypothesis testing
├── ml_preprocessing.py                    # Phase 4: Data preprocessing & encoding
├── ml_logistic_regression.py              # Phase 4: Logistic Regression baseline
├── ml_random_forest.py                    # Phase 4: Random Forest model
├── ml_xgboost.py                          # Phase 4: XGBoost model
├── ml_models.ipynb                        # Phase 4: Full executed ML notebook
├── requirements.txt                       # Python dependencies
└── project_proposal DSA 210.pdf          # Project proposal
```

---

## How to Reproduce

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install scikit-learn xgboost shap
```

### 2. Run Phase 3 — EDA & Hypothesis Testing
```bash
python eda_hypothesis.py
```

### 3. Run Phase 4 — ML Pipeline (in order)
```bash
python ml_preprocessing.py
python ml_logistic_regression.py
python ml_random_forest.py
python ml_xgboost.py
```

### 4. Or run the full executed notebook
```bash
python -m nbconvert --to notebook --execute ml_models.ipynb --output ml_models.ipynb
```

---

## Key Findings

### EDA & Hypothesis Testing
- Exit rate is **14.7%** across 50,000 startups.
- Hypothesis tests (Chi-Square, Mann-Whitney U) found **no statistically significant** difference in exit rates based on investor tier, revenue, funding amount, or industry alone (all p > 0.05).
- This suggests exit outcomes are not driven by any single feature in isolation.

### Machine Learning Results
| Model | ROC-AUC | F1 (Exited) | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.52 | 0.23 | 0.51 |
| Random Forest | 0.50 | 0.17 | 0.71 |
| XGBoost | 0.49 | 0.19 | 0.66 |

- All models performed near random (AUC ≈ 0.50), confirming that **exit success is highly stochastic** and not easily predicted from structured financial data alone.
- Feature importance analysis consistently showed **financial fundamentals** (valuation, revenue, funding) as more important than **investor prestige** features.

### Answer to Research Question
> **Internal business fundamentals are more predictive of exit success than investor prestige**, but neither set of features provides strong predictive power in isolation — suggesting that startup exits depend heavily on factors beyond structured financial metrics.

---

## AI Tool Disclosure
In accordance with the course's academic integrity policy, the use of AI tools in this project is disclosed below.

**Tool used:** Claude (Anthropic) — accessed via claude.ai

**How it was used:**
- Debugging Python errors (e.g. fixing seaborn palette issues, pandas type errors)
- Git command guidance (committing, tagging, pushing)
- Code structure suggestions for the ML pipeline
- Drafting and formatting the final report based on results I generated

**Example prompts used:**
- *"I am getting this error in my seaborn boxplot: ValueError: The palette dictionary is missing keys. How do I fix it?"*
- *"Write me a Python function that computes an investor prestige score based on their historical exit rate in the dataset"*
- *"How do I handle class imbalance in XGBoost? My target variable has a 5.8:1 ratio"*
- *"I ran logistic regression, random forest and XGBoost and all got ROC-AUC near 0.50. What does this mean in the context of my research question?"*
- *"Help me write the findings section of my report based on these hypothesis test results and feature importance plots"*
- *"What git commands do I need to create and push a tag called milestone1?"*

**What was my own work:**
- Research question and hypothesis design — motivated by my personal experience as a startup founder
- All methodological decisions (model selection, feature choices, hypothesis tests)
- All code was executed and verified locally on my own machine
- Interpretation of findings and connection to real-world startup dynamics

Claude was used as a coding and writing assistant. All intellectual decisions, research direction, and interpretation of results are my own.

---

## Author
**Arad Zeinalifarid**
**Course:** DSA 210 - Introduction to Data Science
**Institution:** Sabanci University
