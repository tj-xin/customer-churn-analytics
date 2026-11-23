# Kaggle Submission History

## Competition

**Name:** Playground Series S6E3 — Predict Customer Churn

**Evaluation Metric:** ROC-AUC

**Competition Status:** Completed — Late Submission

## Submission Records

| Version | Model | Local AUC | Public Score | Private Score |
|---|---|---:|---:|---:|
| v1 | LightGBM | 0.91600 | 0.91295 | 0.91415 |

## Submission v1 — LightGBM

**Model Configuration**

- `n_estimators`: 300
- `learning_rate`: 0.05
- `num_leaves`: 31
- `random_state`: 42
- `n_jobs`: -1

**Training Data**

- Full labeled training dataset: 594,194 records
- Test dataset: 254,655 records

**Submission File:** `submission.csv`

**Submission Status:** Complete (after deadline)

**Results**

- Local validation ROC-AUC: 0.91600
- Kaggle Public ROC-AUC: 0.91295
- Kaggle Private ROC-AUC: 0.91415

**Notes**

The final model was retrained using the complete labeled training dataset.

Local validation and Kaggle scores were obtained from different datasets and model fits.

This submission serves as the first recorded Kaggle result for the project.
