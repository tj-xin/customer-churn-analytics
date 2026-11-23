# Data Dictionary

## 1. Dataset Overview

**Project:** Customer Churn Analytics & Prediction

**Source:** Kaggle Playground Series S6E3 — Predict Customer Churn (2026)

**Task:** Binary classification — predicting customer churn.

| Dataset | Records | Columns |
|---|---:|---:|
| Training set | 594,194 | 21 |
| Test set | 254,655 | 20 |

The dataset contains customer demographics, subscribed services, contract information, billing information, and churn labels.

The competition dataset is synthetic and should not be represented as real enterprise customer records.

## 2. Feature Dictionary

### 2.1 Customer Information（客户基本信息）

| Feature | Description | 中文含义 | Data Type |
|---|---|---|---|
| `id` | Record identifier | 客户记录标识符 | int64 |
| `gender` | Customer gender | 性别 | categorical |
| `SeniorCitizen` | Whether the customer is a senior citizen (1 = Yes, 0 = No) | 是否为老年客户 | int64 |
| `Partner` | Whether the customer has a partner | 是否有伴侣 | categorical |
| `Dependents` | Whether the customer has dependents | 是否有受抚养人 | categorical |
| `tenure` | Number of months the customer has stayed with the provider | 在网时长（月） | int64 |

### 2.2 Service Information（服务订阅信息）

| Feature | Description | 中文含义 | Data Type |
|---|---|---|---|
| `PhoneService` | Whether the customer subscribes to phone service | 是否订购电话服务 | categorical |
| `MultipleLines` | Whether the customer has multiple phone lines | 是否拥有多条电话线路 | categorical |
| `InternetService` | Type of internet service | 互联网服务类型 | categorical |
| `OnlineSecurity` | Whether online security service is subscribed | 是否订购网络安全服务 | categorical |
| `OnlineBackup` | Whether online backup service is subscribed | 是否订购在线备份服务 | categorical |
| `DeviceProtection` | Whether device protection service is subscribed | 是否订购设备保护服务 | categorical |
| `TechSupport` | Whether technical support service is subscribed | 是否订购技术支持服务 | categorical |
| `StreamingTV` | Whether streaming TV service is subscribed | 是否订购流媒体电视服务 | categorical |
| `StreamingMovies` | Whether streaming movie service is subscribed | 是否订购流媒体电影服务 | categorical |

### 2.3 Contract and Billing（合同与费用信息）

| Feature | Description | 中文含义 | Data Type |
|---|---|---|---|
| `Contract` | Customer contract type | 合同类型 | categorical |
| `PaperlessBilling` | Whether paperless billing is enabled | 是否使用电子账单 | categorical |
| `PaymentMethod` | Customer payment method | 支付方式 | categorical |
| `MonthlyCharges` | Monthly service charges | 每月服务费用 | float64 |
| `TotalCharges` | Accumulated service charges | 累计服务费用 | float64 |

### 2.4 Target Variable（目标变量）

| Feature | Description | 中文含义 | Data Type |
|---|---|---|---|
| `Churn` | Whether the customer has churned | 客户是否流失 | categorical |

Target encoding used in modeling:

- `Yes` → 1 (Churn)
- `No` → 0 (No Churn)

The Kaggle test dataset does not contain the `Churn` column.

## 3. Categorical Feature Values

The following values were observed in the training dataset.

| Feature | Observed Values |
|---|---|
| `gender` | Male, Female |
| `SeniorCitizen` | 0, 1 |
| `Partner` | Yes, No |
| `Dependents` | Yes, No |
| `PhoneService` | Yes, No |
| `MultipleLines` | Yes, No, No phone service |
| `InternetService` | DSL, Fiber optic, No |
| `OnlineSecurity` | Yes, No, No internet service |
| `OnlineBackup` | Yes, No, No internet service |
| `DeviceProtection` | Yes, No, No internet service |
| `TechSupport` | Yes, No, No internet service |
| `StreamingTV` | Yes, No, No internet service |
| `StreamingMovies` | Yes, No, No internet service |
| `Contract` | Month-to-month, One year, Two year |
| `PaperlessBilling` | Yes, No |
| `PaymentMethod` | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `Churn` | Yes, No |

**Important:** `No internet service` and `No phone service` are distinct categorical values. They should not automatically be treated as missing data.

## 4. Data Quality Summary

Initial exploration of the training dataset found:

- Total records: 594,194
- Missing values: None
- Duplicate records: None
- Unique `id` values: Yes
- Overall churn rate: approximately 22.52%

The data quality checks refer to the labeled training dataset.

## 5. Modeling Notes

- `id` is excluded from model features.
- `Churn` is the binary prediction target.
- The remaining 19 columns are used as predictive features.
- `SeniorCitizen` is stored as an integer but represents a binary categorical attribute.
- LightGBM uses native categorical feature handling for the categorical variables.
- The test dataset must use the same feature columns and categorical definitions as the training dataset.

## 6. Interpretation Notes

1. `tenure` represents customer tenure in months, not internet usage duration.
2. `MonthlyCharges` and `TotalCharges` are monetary variables; the original dataset does not establish a specific currency.
3. `TotalCharges` is related to tenure and recurring charges but should not automatically be assumed to equal `tenure × MonthlyCharges`.
4. Associations between customer characteristics and churn do not establish causal relationships.
5. Findings from synthetic competition data may not generalize directly to real telecommunications customers.
