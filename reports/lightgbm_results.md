# LightGBM Model — Experiment Results

## 1. Experiment Overview

**Project:** Customer Churn Analytics & Prediction

**Dataset:** Kaggle Playground Series S6E3 — Predict Customer Churn

**Model:** LightGBM Classifier

**Objective:** Improve customer churn prediction performance and identify high-risk customers under limited marketing budgets.

## 2. Data and Validation

| Item | Value |
|---|---|
| Training samples | 475,355 |
| Validation samples | 118,839 |
| Features | 19 |
| Training ratio | 80% |
| Validation ratio | 20% |
| Validation strategy | Stratified random split |
| Random state | 42 |
| Churn rate | Approximately 22.52% |

**Preprocessing:**
- Excluded `id` and `Churn` from the feature matrix.
- Encoded `Churn` as a binary target (Yes = 1, No = 0).
- Converted 15 categorical features to Pandas `category` dtype.
- Used LightGBM native categorical feature handling.

## 3. Model Configuration

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)
```

These are initial model parameters, not the result of systematic hyperparameter optimization.

## 4. Model Performance

| Metric | Logistic Regression | LightGBM |
|---|---:|---:|
| Train ROC-AUC | 0.9078 | 0.9183 |
| Validation ROC-AUC | 0.9084 | 0.9160 |
| Train–Validation AUC Gap | -0.0006 | 0.0024 |

LightGBM improved validation ROC-AUC by 0.0076, equivalent to 0.76 percentage points.

No substantial train–validation performance gap was observed.

## 5. Budget-Constrained Customer Targeting

Customers were ranked by predicted churn probability. The highest-risk customers were selected according to simulated marketing contact budgets.

| Budget | Contacted | True Churn | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|---:|
| Top 10% | 11,883 | 9,859 | 0.8297 | 0.3684 | 3.6841 |
| Top 20% | 23,767 | 16,958 | 0.7135 | 0.6336 | 3.1683 |
| Top 30% | 35,651 | 21,804 | 0.6116 | 0.8147 | 2.7157 |

### Comparison with Logistic Regression

| Budget | LR Recall@K | LightGBM Recall@K |
|---|---:|---:|
| Top 10% | 0.3565 | 0.3684 |
| Top 20% | 0.6212 | 0.6336 |
| Top 30% | 0.8039 | 0.8147 |

At the Top 20% contact budget, LightGBM identified 332 additional true churn customers compared with Logistic Regression, without increasing the number of customers contacted.

These results measure predictive targeting performance, not the actual effect of retention interventions.

## 6. Feature Importance

Three feature importance methods were examined:

- **Split Importance:** Frequency of feature usage in tree splits.
- **Gain Importance:** Cumulative split gain attributed to each feature.
- **Permutation Importance:** Reduction in validation ROC-AUC after randomly permuting a feature.

### Gain Importance

| Feature | Gain Share |
|---|---:|
| Contract | 49.29% |
| InternetService | 13.40% |
| PaymentMethod | 12.87% |
| tenure | 12.18% |
| MonthlyCharges | 3.08% |
| TotalCharges | 1.98% |

### Permutation Importance

Computed using 10,000 randomly sampled validation observations and three permutation repetitions.

| Feature | Mean AUC Drop | Std |
|---|---:|---:|
| Contract | 0.0945 | 0.0019 |
| tenure | 0.0638 | 0.0029 |
| TotalCharges | 0.0232 | 0.0003 |
| InternetService | 0.0153 | 0.0006 |
| PaymentMethod | 0.0071 | 0.0005 |
| MonthlyCharges | 0.0043 | 0.0007 |

### Interpretation

Contract was ranked first by both Gain and Permutation Importance.

Tenure and TotalCharges also provided substantial predictive information.

Differences between Split, Gain, and Permutation rankings demonstrate that feature importance depends on the measurement method.

Feature importance does not establish causal effects. Correlated predictors can also affect the interpretation of individual feature contributions.

## 7. Limitations

1. The Kaggle Playground dataset is competition data and should not be presented as real enterprise operational data.
2. Results were obtained using one stratified train–validation split.
3. Statistical significance of the model improvement has not yet been tested.
4. Permutation Importance may be affected by correlated features.
5. Marketing budgets are hypothetical scenarios.
6. No randomized retention intervention data is available; actual churn reduction and financial returns cannot be established from these results.

## 8. Next Steps

- Conduct customer churn risk segmentation.
- Develop hypothetical marketing cost-benefit scenarios.
- Evaluate whether further model optimization is worthwhile.
- Prepare Kaggle submission and reproducible project documentation.
