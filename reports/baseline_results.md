# Logistic Regression Baseline — Experiment Record

## 1. Dataset and validation

- Source: Kaggle Playground Series S6E3, Predict Customer Churn (synthetic competition dataset).
- Training file: `train.csv`, 594,194 rows and 21 columns, including `id` and `Churn`.
- Kaggle test file: `test.csv`, 254,655 rows and 20 columns, without `Churn`.
- Target: `Churn` (`Yes` = 1; `No` = 0).
- Validation: stratified random 80/20 split (training: 475,355; validation: 118,839).
- Positive class prevalence: approximately 22.52% in both subsets.
- Random seed and exact split/preprocessing parameters: **to be copied from the notebook before finalizing reproducibility**.

## 2. Model and preprocessing

- Model: scikit-learn Logistic Regression baseline.
- Numeric features: `tenure`, `MonthlyCharges`, `TotalCharges`.
- Binary feature: `SeniorCitizen`.
- Categorical features: 15 columns.
- Identifier `id` excluded from predictors.
- Preprocessing: numeric scaling and categorical one-hot encoding, as implemented in `02_baseline_model.ipynb`.
- Exact estimator parameters, imputation settings (if any), and library versions: **verify in notebook/environment**.

## 3. Validation results

| Metric | Value |
|---|---:|
| Train ROC-AUC | 0.9078 |
| Validation ROC-AUC | 0.9084 |
| Average Precision (AP) | 0.7271 |
| Precision at threshold 0.5 | 0.6839 |
| Recall at threshold 0.5 | 0.6577 |
| F1 at threshold 0.5 | 0.6705 |

Confusion matrix at threshold 0.5 (rows = actual 0/1; columns = predicted 0/1):

```text
[[83941,  8135],
 [ 9162, 17601]]
```

Train and validation AUC differ by -0.0006, showing no obvious train/validation performance gap. This alone does not rule out leakage or distribution shift.

## 4. Threshold analysis

| Threshold | Precision | Recall | F1 | Predicted churn |
|---:|---:|---:|---:|---:|
| 0.2 | 0.5317 | 0.8935 | 0.6667 | 44,974 |
| 0.3 | 0.5831 | 0.8311 | 0.6854 | 38,142 |
| 0.4 | 0.6310 | 0.7566 | 0.6881 | 32,092 |
| 0.5 | 0.6839 | 0.6577 | 0.6705 | 25,736 |
| 0.6 | 0.7426 | 0.5211 | 0.6125 | 18,781 |
| 0.7 | 0.8185 | 0.3300 | 0.4704 | 10,791 |

Lowering the threshold increases recall and the number of customers flagged, while also increasing the workload. No business-optimal threshold has been established.

## 5. Top-K targeting under simulated contact budgets

| Contact budget (% of validation customers) | Contacted | Actual churners in list | Precision@K | Recall@K | Lift@K (approx.) |
|---:|---:|---:|---:|---:|---:|
| 10% | 11,883 | 9,542 | 0.8030 | 0.3565 | 3.57 |
| 20% | 23,767 | 16,626 | 0.6995 | 0.6212 | 3.11 |
| 30% | 35,651 | 21,514 | 0.6035 | 0.8039 | 2.68 |

Lift@K = Precision@K / overall validation churn rate (approximately 0.2252).

For a simulated 20% contact capacity, the highest-scored 20% of customers contain approximately 62.12% of observed churners. This is **risk targeting**, not a measured retention intervention effect.

## 6. Interpretation, limitations, and next steps

- Baseline discrimination is strong on the current holdout split (ROC-AUC = 0.9084).
- The business objective is to improve recall subject to a limited customer-contact budget; compare models at the same Top-K budget.
- ROC-AUC assesses ranking, not probability calibration, intervention effectiveness, or financial returns.
- This is a synthetic Kaggle dataset; findings should not be represented as actual company outcomes.
- Do not claim that contacting flagged customers prevents churn without intervention evidence.
- Next: train LightGBM on the **same split**, compare ROC-AUC, AP, Precision@K, Recall@K and Lift@K; preserve the baseline as a reproducible reference.

## 7. Reproduction checklist

- [ ] Record the split `random_state`, stratification, and exact feature list.
- [ ] Confirm all preprocessing was fitted on the training partition only.
- [ ] Record Logistic Regression parameters and package versions.
- [ ] Rerun the notebook from a clean kernel to reproduce the values above.
