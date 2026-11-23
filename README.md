# Customer Churn Analytics & Prediction

[English](README.md) | [简体中文](README_CN.md)

**An end-to-end data analytics and machine learning project for customer churn prediction, risk segmentation, and business-oriented retention analysis.**

This project combines statistical analysis, interpretable machine learning, gradient boosting, and hypothetical retention campaign simulations to explore how predictive analytics can support customer retention decisions.

**Technologies:** Python · Pandas · NumPy · SQL · Scikit-learn · LightGBM · Matplotlib · Seaborn · Git

---

## 1. Project Highlights

| Key Metric | Result |
|---|---:|
| Training Dataset | 594,194 records |
| Local Validation ROC-AUC | **0.9160** |
| Kaggle Public ROC-AUC | **0.91295** |
| Kaggle Private ROC-AUC | **0.91415** |
| LightGBM PR-AUC (Average Precision) | **0.7531** |
| LightGBM Brier Score | **0.0954** |
| Recall@Top 20% | **63.36%** |

**Key achievements:**

- Developed a complete analytical workflow from data exploration to Kaggle submission.
- Compared an interpretable Logistic Regression baseline with a LightGBM classifier.
- Improved validation ROC-AUC from 0.9084 to 0.9160.
- Identified 63.36% of actual churners by targeting the highest-risk 20% of validation customers.
- Analyzed customer risk segments and evaluated hypothetical retention campaign economics.
- Organized reusable Python modules, validation scripts, experiment reports, and dataset documentation.

**Kaggle submission status:** Completed after the competition deadline. The reported scores are official evaluation results, not an in-competition ranking.

---

## 2. Business Background and Objectives

Customer churn refers to the loss of existing customers. For subscription-based businesses, understanding churn risk can help prioritize customer engagement and retention resources.

However, a useful churn analytics project should address more than prediction accuracy.

This project investigates three connected questions:

**1. Customer Analytics**

Which customer characteristics and service patterns are associated with higher churn rates?

**2. Predictive Modeling**

How accurately can machine learning models distinguish customers who churn from those who remain?

**3. Business Decision Support**

Given a limited retention budget, how can predicted churn probabilities help prioritize customers, and what might the financial implications be under explicitly stated assumptions?

The project emphasizes analytical reasoning, reproducibility, and business interpretation rather than leaderboard optimization alone.

---

## 3. Dataset

### 3.1 Data Source

**Competition:** Kaggle Playground Series — Season 6, Episode 3: Predict Customer Churn (2026)

The dataset contains synthetic customer records with demographic information, service subscriptions, billing characteristics, contract details, and churn labels.

| Dataset | Rows | Columns |
|---|---:|---:|
| Training | 594,194 | 21 |
| Test | 254,655 | 20 |

The training dataset contains:

- 19 predictive features
- 1 record identifier (`id`)
- 1 target variable (`Churn`)

The target is binary:

- `Yes`: Customer churned
- `No`: Customer did not churn

The observed churn rate in the training dataset is approximately **22.52%**.

### 3.2 Feature Categories

| Category | Representative Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Customer History | `tenure` |
| Service Subscriptions | `PhoneService`, `InternetService`, `TechSupport`, `OnlineSecurity` |
| Contract and Billing | `Contract`, `PaymentMethod`, `PaperlessBilling` |
| Financial Information | `MonthlyCharges`, `TotalCharges` |

For complete field descriptions, see [Data Dictionary](docs/data_dictionary.md).

### 3.3 Data Quality

Initial data exploration found:

- No missing values in the labeled training dataset
- No duplicate training records
- Unique record identifiers
- A moderately imbalanced binary target distribution

**Important:** This is a synthetic Kaggle competition dataset, not a real company's proprietary customer database. Its distributions and predictive relationships may not generalize to real operational settings.

---

## 4. Exploratory Data Analysis

Exploratory data analysis (EDA) was conducted to investigate customer behavior, churn distributions, and relationships among predictive features.

### 4.1 Churn by Contract Type

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | 42.05% |
| One year | 5.76% |
| Two year | 1.00% |

Month-to-month customers exhibited substantially higher observed churn rates than customers with longer-term contracts.

To investigate whether differences in customer tenure could explain part of this pattern, churn rates were also standardized across tenure groups.

| Contract Type | Tenure-Standardized Churn Rate |
|---|---:|
| Month-to-month | 35.15% |
| One year | 5.12% |
| Two year | 0.67% |

The differences remained substantial after standardization.

These findings indicate a strong statistical association between contract type and churn. They do not establish that changing a customer's contract would causally reduce churn.

### 4.2 Tenure and Billing Characteristics

Additional findings included:

- Customer tenure and total charges were strongly correlated (Pearson's r ≈ 0.7697).
- Churn rates differed across monthly charge groups.
- Internet service type was associated with different customer billing patterns and churn rates.
- Contract type, tenure, and service characteristics contained potentially useful predictive information.

These findings motivated the use of multivariate predictive models to investigate customer churn more systematically.

**Notebook:** [Exploratory Data Analysis](notebooks/01_data_overview.ipynb)

---

## 5. Machine Learning Methodology

### 5.1 Validation Strategy

A stratified holdout validation strategy was used.

| Setting | Configuration |
|---|---|
| Training proportion | 80% |
| Validation proportion | 20% |
| Training subset | 475,355 records |
| Validation subset | 118,839 records |
| Random state | 42 |
| Primary evaluation metric | ROC-AUC |

Stratification preserved the churn class distribution in both subsets.

The Kaggle test dataset was not used for local model evaluation.

### 5.2 Logistic Regression Baseline

Logistic Regression was selected as the baseline model because it provides:

- An interpretable statistical framework for binary classification
- A benchmark for evaluating more flexible models
- A connection between statistical modeling and practical machine learning

The baseline achieved a validation ROC-AUC of **0.9084**.

**Notebook:** [Logistic Regression Baseline](notebooks/02_baseline_model.ipynb)

### 5.3 LightGBM Model

LightGBM was used to capture nonlinear relationships and feature interactions.

The model employed native categorical feature handling.

Main hyperparameters:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `random_state` | 42 |
| `n_jobs` | -1 |

The project prioritizes a transparent modeling workflow over extensive hyperparameter searching.

**Notebook:** [LightGBM Modeling](notebooks/03_lightgbm_model.ipynb)

---

## 6. Model Evaluation

### 6.1 Model Comparison

| Metric | Logistic Regression | LightGBM |
|---|---:|---:|
| Training ROC-AUC | 0.9078 | 0.9183 |
| Validation ROC-AUC | 0.9084 | **0.9160** |
| PR-AUC (Average Precision) | 0.7271 | **0.7531** |

LightGBM improved validation ROC-AUC by approximately **0.0076** in absolute terms.

Its training-validation ROC-AUC gap was approximately 0.0024, suggesting limited apparent overfitting under this holdout split.

This does not establish performance stability across different time periods or real-world populations.

### 6.2 Probability Quality

The LightGBM model achieved a validation **Brier Score of 0.0954**.

Probability calibration was also examined using calibration curves.

These analyses matter because customer retention decisions may depend on estimated churn probabilities, not only binary predictions.

### 6.3 Kaggle Submission

After local model evaluation, the LightGBM classifier was retrained using all 594,194 labeled training records.

Predicted churn probabilities were generated for the 254,655 Kaggle test records.

| Evaluation Dataset | ROC-AUC |
|---|---:|
| Local Validation | 0.91600 |
| Kaggle Public | 0.91295 |
| Kaggle Private | **0.91415** |

**Submission status:** Complete (after deadline).

The local validation and Kaggle results come from different evaluation datasets. The final submission also uses a model refitted on the full labeled training data.

These results demonstrate predictive performance on the competition data, but should not be interpreted as evidence of real-world deployment performance.

See [Kaggle Submission History](kaggle/submission_history.md) for details.

---

## 7. Model Interpretation

Feature importance was investigated using LightGBM gain-based importance and permutation importance.

Important predictive features included:

- `Contract`
- `tenure`
- `InternetService`
- `PaymentMethod`
- `TotalCharges`
- `MonthlyCharges`

Contract type showed particularly strong predictive importance.

The analysis also highlighted differences between importance measures. For example, features that are frequently used for tree splits are not necessarily the same features that contribute the largest improvements in predictive performance.

**Interpretation considerations:**

- Correlated features can share predictive information.
- Importance rankings depend on the fitted model and the evaluation method.
- Predictive importance does not establish causal influence.

The findings are therefore interpreted as model-based associations rather than intervention effects.

---

## 8. Business-Oriented Churn Analysis

### 8.1 Customer Risk Segmentation

Customers in the validation dataset were divided into five approximately equal-sized groups based on predicted churn probability.

| Risk Segment | Observed Churn Rate |
|---|---:|
| Very Low | 0.21% |
| Low | 1.24% |
| Medium | 7.51% |
| High | 32.30% |
| Very High | 71.35% |

The results demonstrate strong separation between low-risk and high-risk customers.

This segmentation can support differentiated customer retention strategies, subject to business constraints and further validation.

### 8.2 Top-K Targeting Performance

In practice, a company may only have enough resources to contact a limited proportion of its customers.

Top-K evaluation measures the effectiveness of prioritizing customers by predicted churn probability.

| Contact Budget | Actual Churners Captured | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|
| Top 10% | 9,859 | 82.97% | 36.84% | 3.68 |
| Top 20% | 16,958 | 71.35% | 63.36% | 3.17 |
| Top 30% | 21,804 | 61.16% | 81.47% | 2.72 |

**Key finding:** Targeting the highest-risk 20% of customers captured approximately 63.36% of actual churners in the validation dataset.

This illustrates how ranking performance can inform customer targeting under limited campaign budgets.

### 8.3 Hypothetical Retention Profit Simulation

A simplified financial model was developed to explore how targeting decisions might affect expected campaign profitability.

The simulation uses:

\[
E[\mathrm{Profit}(K)] = TP@K \times q \times V - K \times C
\]

Where:

- \(TP@K\): Number of actual churners among the selected customers
- \(q\): Assumed intervention success rate
- \(V\): Assumed value of successfully retaining a customer
- \(K\): Number of contacted customers
- \(C\): Assumed cost per contact

**Simulation assumptions:**

| Parameter | Assumed Value |
|---|---:|
| Cost per contact | CNY 10 |
| Value per retained customer | CNY 200 |
| Retention success rates | 10%, 20%, 30% |

Simulated expected net profits:

| Contact Budget | 10% Success | 20% Success | 30% Success |
|---|---:|---:|---:|
| Top 10% | CNY 78,350 | CNY 275,530 | CNY 472,710 |
| Top 20% | CNY 101,490 | CNY 440,650 | CNY 779,810 |
| Top 30% | CNY 79,570 | CNY 515,650 | CNY 951,730 |

At a hypothetical 20% intervention success rate, LightGBM identified 332 more actual churners than Logistic Regression within the top 20% contact budget.

Under the same financial assumptions, this corresponds to **CNY 13,280 in additional simulated expected net profit**.

### 8.4 Business Interpretation and Limitations

These results suggest that predictive churn scores can help prioritize customers for further retention analysis.

However, identifying a customer who is likely to churn is not the same as identifying a customer whose behavior can be changed through an intervention.

The simulation assumes intervention effectiveness and does not estimate causal treatment effects.

**All financial results are hypothetical. No actual marketing intervention or realized financial return is claimed.**

**Notebook:** [Business Analysis](notebooks/04_business_analysis.ipynb)

---

## 9. Project Structure

```text
customer-churn-analytics/
├── data/                         # Local Kaggle datasets (Git-ignored)
├── docs/
│   └── data_dictionary.md
├── kaggle/
│   └── submission_history.md
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_lightgbm_model.ipynb
│   ├── 04_business_analysis.ipynb
│   └── 05_final_submission.ipynb
├── reports/
│   ├── baseline_results.md
│   ├── lightgbm_results.md
│   └── business_analysis.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluation.py
│   └── business.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_evaluation.py
│   ├── test_business.py
│   └── check_real_results.py
├── .gitignore
├── environment.yml
├── README.md
└── README_CN.md
```

### Directory Responsibilities

| Directory | Purpose |
|---|---|
| `data/` | Raw competition data stored locally |
| `docs/` | Dataset documentation |
| `kaggle/` | Competition submission records |
| `notebooks/` | Exploratory analysis, modeling, and submission workflows |
| `reports/` | Experimental results and business analysis |
| `src/` | Reusable data processing, evaluation, and business functions |
| `tests/` | Module checks and result verification |

The repository separates exploratory workflows from reusable Python functions and documented results.

---

## 10. Reproducibility

### 10.1 Clone the Repository

```bash
git clone https://github.com/tj-xin/customer-churn-analytics.git
cd customer-churn-analytics
```

### 10.2 Set Up the Environment

The project uses Python 3.12 and provides a Conda environment configuration.

```bash
conda env create -f environment.yml
conda activate churn-analysis
```

The current environment file includes the core data processing and notebook dependencies.

Install additional modeling and SQL packages:

```bash
python -m pip install scikit-learn lightgbm duckdb
```

These packages should eventually be added to `environment.yml` to consolidate dependency management.

### 10.3 Download the Dataset

Obtain the original competition files from Kaggle and place them in the following local directory:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Raw competition files are excluded from Git version control.

### 10.4 Run the Notebooks

Launch JupyterLab from the project root:

```bash
jupyter lab
```

Recommended notebook execution order:

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `01_data_overview.ipynb` | Data understanding and EDA |
| 2 | `02_baseline_model.ipynb` | Logistic Regression baseline |
| 3 | `03_lightgbm_model.ipynb` | LightGBM modeling and evaluation |
| 4 | `04_business_analysis.ipynb` | Risk segmentation and profit simulation |
| 5 | `05_final_submission.ipynb` | Final model training and Kaggle predictions |

Some downstream notebooks require artifacts generated by earlier steps.

The notebooks represent the original experimental workflow, while the `src/` package contains reusable functions developed during the project.

### 10.5 Run Module Checks

Run the available test scripts from the repository root:

```bash
python -m tests.test_data
python -m tests.test_evaluation
python -m tests.test_business
```

These scripts check data preparation, model evaluation, and business simulation functions.

### 10.6 Verify Saved Validation Results

When `reports/validation_predictions.csv` is available locally, run:

```bash
python -m tests.check_real_results
```

The verification script checks the saved LightGBM predictions against the documented evaluation results.

The prediction file is not committed to GitHub and must be generated or supplied locally.

This verification checks previously generated predictions; it does not replace retraining the original model.

---

## 11. Limitations

Several limitations should be considered when interpreting the results.

**Synthetic competition data**

The dataset does not necessarily reflect the distributions, operational processes, or customer behavior of a real telecommunications company.

**Single holdout validation**

A stratified train-validation split provides an initial estimate of generalization performance, but does not measure temporal stability or robustness across organizations.

**Predictive rather than causal modeling**

Relationships between contract types, services, billing characteristics, and churn should not be interpreted as causal effects.

**Hypothetical business assumptions**

Retention costs, customer values, and intervention success rates were assumed for simulation purposes.

**No production deployment**

The project focuses on offline analytics, model development, evaluation, and business interpretation rather than production infrastructure.

---

## 12. Key Takeaways

This project demonstrates a complete workflow connecting statistical analysis, machine learning, and business decision support.

The main outcomes are:

1. Analyzed **594,194 labeled customer records** to investigate churn patterns and customer characteristics.
2. Built Logistic Regression and LightGBM models using a consistent stratified validation strategy.
3. Achieved **0.9160 local validation ROC-AUC** and **0.91415 Kaggle Private ROC-AUC**.
4. Captured **63.36% of actual churners** within the highest-risk 20% of validation customers.
5. Developed risk segmentation and hypothetical financial simulations to evaluate targeting strategies.
6. Organized the project into documented notebooks, reusable Python modules, and validation scripts.

The project illustrates how predictive modeling can support customer analytics while emphasizing reproducibility, transparent assumptions, and responsible business interpretation.

---

## 13. References and Acknowledgments

**Dataset:** Kaggle Playground Series, Season 6 Episode 3 — Predict Customer Churn (2026).

**Repository:** https://github.com/tj-xin/customer-churn-analytics

This is an independent educational and portfolio project. It is not affiliated with Kaggle or any telecommunications company.

The competition dataset is synthetic, and all reported model results are based on completed experiments. Financial simulations are illustrative and should not be interpreted as realized business outcomes.