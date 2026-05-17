# Final Report: Evaluating the Impact of Startup Fundamentals and Investor Prestige on Exit Success

**Author:** Arad Zeinalifarid
**Course:** DSA 210 - Introduction to Data Science
**Institution:** Sabanci University
**Date:** May 2026

---

## 1. Motivation

My interest in this project is personal. I have founded a startup, gone through the fundraising process, and experienced firsthand the weight that investor reputation carries in early-stage conversations. Founders are constantly told that *who* backs you matters as much as *what* you build — that landing a top-tier VC signals quality, opens doors, and dramatically increases your odds of a successful exit. At the same time, operators and advisors I have worked with argue the opposite: that fundamentals always win in the long run, and that investor prestige is a surface-level signal that fades once a company has to perform.

This tension between two competing narratives — **network prestige vs. business fundamentals** — is something I wanted to test with real data. As someone building in this ecosystem, the answer has practical implications: where should a founder focus their energy? Is chasing a brand-name investor worth the dilution and time, or is relentless focus on revenue and efficiency the better path to an exit?

This project gave me the opportunity to approach that question rigorously, applying the data science methods learned in DSA 210 to a domain I care deeply about.

---

## 2. Data Source

The dataset used is the **Startup Valuation Dataset (2010–2024)**, sourced from Kaggle. It contains 50,000 unique startup records representing global funding activity across 15 years and 7 major industries (AI/ML, Fintech, SaaS, HealthTech, EdTech, E-commerce, and Marketplace).

**Key features include:**
- **Metadata:** Startup name, founding year, country, region, industry
- **Financials:** Funding round type, funding amount (USD), estimated revenue (USD), estimated valuation (USD)
- **Operational:** Employee count, company tags
- **Investor data:** Lead investor name, co-investor network
- **Target variable:** `exited` — binary (True/False), representing whether the startup achieved a liquidity event (IPO or acquisition)

The dataset was collected by downloading the structured CSV file from Kaggle and loading it directly into Python for processing. No web scraping or API calls were required. The dataset required no external joining, but significant **feature engineering** was performed to enrich the raw columns into meaningful analytical variables, as described in the next section.

**Dataset characteristics:**
- 50,000 records, 17 raw columns
- Exit rate: 14.7% (7,333 exits out of 50,000)
- Class imbalance ratio: 5.8:1 (not exited to exited)
- No missing values in core features; 45,100 null values in `exit_type` (expected — non-exited startups have no exit type)

---

## 3. Data Analysis

### 3.1 Data Cleaning
The raw dataset was first cleaned by standardizing column names, converting the boolean `exited` column to binary integer format, and filling null values in `exit_type` with "None" for non-exited startups. No duplicate records were found.

### 3.2 Feature Engineering
Five new features were engineered to better capture the constructs of interest:

| Feature | Description | Represents |
|---|---|---|
| `revenue_per_employee` | Estimated revenue / employee count | Operational efficiency |
| `valuation_to_funding` | Valuation / total funding raised | Capital efficiency |
| `company_age` | 2024 − founding year | Startup maturity |
| `investor_prestige_score` | Lead investor's historical exit rate within the dataset | Investor prestige (continuous) |
| `is_top_tier_investor` | Binary flag: top 25% by prestige score | Investor prestige (categorical) |

Categorical variables (`industry`, `region`, `funding_round`) were label-encoded for use in machine learning models.

### 3.3 Exploratory Data Analysis
A comprehensive EDA was conducted to understand the structure of the data before modeling:

- **Target distribution:** The exit rate of 14.7% confirms a significant class imbalance, requiring special handling in models.
- **Feature distributions:** Revenue, funding amount, and valuation all follow approximately log-normal distributions, as expected for financial data.
- **Exit rate by industry:** Variation exists across industries, but no single industry shows dramatically higher exit rates.
- **Exit rate by region:** Similar regional variation observed, with no region being a strong standalone predictor.
- **Correlation heatmap:** Moderate correlations between financial variables (revenue, valuation, funding) were observed, but `investor_prestige_score` showed low correlation with `exited`.
- **Investor prestige quintiles:** Exit rates across investor prestige quintiles were nearly uniform, providing early evidence against the prestige hypothesis.

### 3.4 Hypothesis Testing
Four formal hypothesis tests were conducted at α = 0.05:

| Hypothesis | Test | Result | p-value |
|---|---|---|---|
| H1: Top-tier investors → higher exit rate | Chi-Square | Fail to reject H0 | 0.1305 |
| H2: Exited startups have higher revenue | Mann-Whitney U | Fail to reject H0 | 0.4929 |
| H3: Exit rate differs across industries | Chi-Square | Fail to reject H0 | 0.1236 |
| H4: Exited startups have higher funding | Mann-Whitney U | Fail to reject H0 | 0.5553 |

All four tests failed to find statistically significant differences, suggesting that no single feature alone is a reliable predictor of exit success.

### 3.5 Machine Learning Models
Three classification models were trained to predict the binary `exited` outcome using the full 12-feature set. An 80/20 stratified train/test split was used. Class imbalance was handled via `class_weight='balanced'` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost).

**Logistic Regression** served as the linear baseline. It assumes a linear decision boundary and was regularized with C=0.1.

**Random Forest** used 200 trees with max depth of 10, capturing non-linear relationships and feature interactions.

**XGBoost** used 300 gradient boosted trees with learning rate 0.05, subsample 0.8, and scale_pos_weight set to the class imbalance ratio (5.82).

**Results:**

| Model | ROC-AUC | F1 (Exited) | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.521 | 0.23 | 0.51 |
| Random Forest | 0.504 | 0.17 | 0.71 |
| XGBoost | 0.491 | 0.19 | 0.66 |

**Feature importance** (Random Forest and XGBoost consistently agreed):
1. `valuation_to_funding` — highest importance
2. `estimated_valuation_usd`
3. `estimated_revenue_usd`
4. `revenue_per_employee`
5. `funding_amount_usd`
6. `employee_count`
7. `company_age`
8. `investor_prestige_score`
9. `industry_encoded`, `funding_round_encoded`, `region_encoded`
10. `is_top_tier_investor` — lowest importance

---

## 4. Findings

### 4.1 All Models Performed Near Random
The most striking finding is that all three models — including the powerful gradient boosting model XGBoost — achieved ROC-AUC scores near 0.50. This means that the available structured features provide almost no predictive power for startup exit outcomes. This is consistent across linear and non-linear models, ruling out the possibility that the failure is due to model choice.

### 4.2 Fundamentals Outrank Investor Prestige
Despite both performing poorly in absolute terms, **business fundamentals consistently ranked higher than investor prestige** in feature importance across all tree-based models. Valuation, revenue, and capital efficiency ratios were the top predictors, while `investor_prestige_score` and `is_top_tier_investor` ranked near the bottom.

This provides partial evidence in favor of the **fundamentals hypothesis**: to the extent that any signal exists in this data, it comes from internal financial metrics rather than who backed the company.

### 4.3 Hypothesis Tests Confirm No Single Feature Predicts Exit
The four hypothesis tests all failed to reject the null hypothesis. Even revenue — the most direct measure of business health — showed no statistically significant difference between exited and non-exited startups (median revenue of $107M vs $106M respectively). This reinforces the conclusion that exit outcomes are driven by a complex combination of factors, none of which dominates alone.

### 4.4 Answer to the Research Question
> **Internal business fundamentals are more predictive of exit success than investor prestige**, but neither set of features provides sufficient predictive power in isolation. Startup exit outcomes appear highly stochastic and depend on factors beyond what is captured in structured financial data.

This finding resonates with real-world venture capital research, which consistently shows that startup success is difficult to predict even with extensive data — a reflection of the inherently uncertain nature of innovation.

---

## 5. Limitations and Future Work

### Limitations

**Synthetic dataset:** The Kaggle dataset simulates real-world startup data but is not sourced from actual company records. This likely reduces the signal quality and may explain why all models performed near random — a real dataset from Crunchbase or PitchBook would contain richer, messier, but more genuine patterns.

**Investor prestige approximation:** Investor prestige was approximated using within-dataset exit rates, not real-world VC reputation rankings. A startup backed by Sequoia and one backed by an unknown local fund could have the same prestige score if their historical exit rates happen to be similar in this dataset. Using actual VC tier rankings (e.g., from industry databases) would be a more accurate measure.

**Missing features:** Key predictors of startup success are absent from this dataset — team quality, founding team's prior experience, product-market fit signals, customer growth rate, burn rate, and macroeconomic conditions at time of funding. These are precisely the factors that experienced investors use to make decisions.

**Static snapshot:** Each startup is represented by a single record, ignoring the trajectory over time. A startup growing from $1M to $50M revenue tells a very different story than one stuck at $10M for five years — yet both would look similar in a static snapshot.

### Future Work

- **Richer data:** Integrate real-world data from Crunchbase API or PitchBook, including actual VC tier rankings, founding team backgrounds, and multiple funding rounds per company.
- **Time-series modeling:** Represent each startup as a sequence of funding events and financial snapshots, enabling models to learn from growth trajectories rather than static features.
- **Better imbalance handling:** Experiment with SMOTE (Synthetic Minority Oversampling Technique) and cost-sensitive learning to improve minority class (exited) prediction.
- **Survival analysis:** Frame exit prediction as a time-to-event problem using Cox Proportional Hazards models, which naturally handle the temporal nature of startup exits.
- **SHAP analysis:** Apply SHAP (SHapley Additive exPlanations) values for deeper individual-level interpretability of model predictions.

---

## 6. AI Tool Disclosure

In accordance with the course's academic integrity policy, the use of AI tools in this project is disclosed below.

**Tool used:** Claude (Anthropic) — accessed via claude.ai

**How it was used:**
- Assistance with Python syntax and debugging (e.g., fixing seaborn palette errors, resolving pandas type errors)
- Code structure suggestions for the ML pipeline (train/test split, scaling, evaluation metrics)
- Guidance on git commands for committing and tagging submissions
- Drafting and formatting this final report based on results and findings I generated

**What was my own work:**
- The research question and hypothesis design were my own, motivated by my personal experience as a founder
- All methodological decisions (choice of models, features, hypothesis tests, significance level) were made by me
- All code was executed locally on my machine and results were verified by me
- Interpretation of findings and connection to real-world startup dynamics reflects my own understanding and experience

Claude was used as a coding and writing assistant — similar to using Stack Overflow for debugging or Grammarly for writing. All intellectual decisions, research direction, and interpretation of results are my own.

---

*Repository: [https://github.com/Ardznl/-Sabanci-Data-Science-Project-](https://github.com/Ardznl/-Sabanci-Data-Science-Project-)*
