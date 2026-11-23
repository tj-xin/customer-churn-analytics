# Customer Churn — Business Analysis Report

## 1. Objective

Evaluate customer churn risk segmentation and simulate budget-constrained retention targeting strategies using LightGBM predictions.

Dataset: Kaggle Playground Series S6E3 — Predict Customer Churn.

All results are based on a held-out validation set from competition data.

## 2. Risk Segmentation

Customers were divided into five equal-sized groups according to their predicted churn scores.

| Risk Level | Customers | Actual Churn Rate |
|---|---:|---:|
| Very Low | 23,768 | 0.21% |
| Low | 23,768 | 1.24% |
| Medium | 23,767 | 7.51% |
| High | 23,768 | 32.30% |
| Very High | 23,768 | 71.35% |

The model demonstrated strong risk differentiation across customer groups.

## 3. High-Risk Customer Profile

Within month-to-month contract customers, the highest-risk 20% exhibited the following characteristics:

| Indicator | Value |
|---|---:|
| Customers | 11,971 |
| Actual churn rate | 82.82% |
| Median tenure | 5 months |
| Average monthly charges | 86.85 |
| Fiber optic service share | 98.28% |

These characteristics describe the model-identified high-risk segment. They do not establish causal relationships.

## 4. Budget-Constrained Targeting

Under a Top 20% contact budget:

| Metric | Logistic Regression | LightGBM |
|---|---:|---:|
| Customers contacted | 23,767 | 23,767 |
| True churn customers identified | 16,626 | 16,958 |
| Recall@K | 62.12% | 63.36% |

LightGBM identified 332 additional true churn customers at the same contact volume.

## 5. Cost-Benefit Simulation

Assumptions:

- Contact cost: 10 CNY per customer
- Value per successful retention: 200 CNY, excluding contact cost
- Retention success rate: 10%, 20%, or 30%

Expected net benefit:

\[
E[\Pi(K)]=TP@K\times q\times V-K\times C
\]

### LightGBM Simulation Results

| Contact Budget | 10% Success | 20% Success | 30% Success |
|---|---:|---:|---:|
| Top 10% | 78,350 | 275,530 | 472,710 |
| Top 20% | 101,490 | 440,650 | 779,810 |
| Top 30% | 79,570 | 515,650 | 951,730 |

Under the 20% success-rate scenario, the Top 30% contact strategy generated the highest simulated net benefit among the three evaluated budgets.

### Model Comparison

At a Top 20% contact budget and a 20% assumed retention success rate:

- Logistic Regression: 427,370 CNY
- LightGBM: 440,650 CNY
- Additional simulated net benefit: 13,280 CNY

## 6. Limitations

1. The dataset is synthetic competition data, not real enterprise customer records.
2. Retention success rates, contact costs, and customer values are hypothetical.
3. Predicting churn is not equivalent to predicting responsiveness to retention interventions.
4. The simulation assumes a constant retention success rate across targeted churn customers.
5. Only three contact-budget levels were compared.
6. No causal effect of marketing intervention was estimated.

## 7. Business Recommendations

- Prioritize high-risk customers when marketing resources are limited.
- Use model-based ranking rather than treating all month-to-month customers as equally risky.
- Consider customer value, contact costs, and intervention effectiveness when determining targeting budgets.
- Validate retention strategies through controlled experiments before real-world deployment.

## 8. Conclusion

LightGBM improved both predictive discrimination and budget-constrained customer targeting compared with Logistic Regression.

Under the stated hypothetical assumptions, the improvement also translated into higher simulated expected net benefits.

The analysis provides a reproducible framework for connecting churn prediction with business decision-making, while recognizing that actual retention effectiveness requires experimental validation.
