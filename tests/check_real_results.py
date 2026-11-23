import pandas as pd

from src.evaluation import evaluate_classifier, evaluate_top_k
from src.business import segment_customers, simulate_retention_profit


# 1. 读取验证集预测结果
df = pd.read_csv("reports/validation_predictions.csv")

print("数据形状：", df.shape)
print("字段名称：", df.columns.tolist())

# 2. 自动识别真实标签和预测概率
label_candidates = ["Churn", "y_true", "actual", "target"]
prob_candidates = [
    "ChurnProbability", "y_prob",
    "pred_proba", "probability"
]

label_col = next(
    (c for c in label_candidates if c in df.columns), None
)
prob_col = next(
    (c for c in prob_candidates if c in df.columns), None
)

if label_col is None or prob_col is None:
    raise ValueError(
        "无法识别标签列或概率列，请根据上方字段名称修改代码。"
    )

y_true = df[label_col].replace({"Yes": 1, "No": 0}).astype(int)
y_prob = df[prob_col].astype(float)

# 3. 模型评价
metrics = evaluate_classifier(y_true, y_prob)

print("\n=== Model Evaluation ===")
print("ROC-AUC:", round(metrics["roc_auc"], 4))
print("PR-AUC:", round(metrics["pr_auc"], 4))
print("Brier Score:", round(metrics["brier_score"], 4))

# 4. Top-K 评价
top_k = evaluate_top_k(y_true, y_prob)

print("\n=== Top-K Evaluation ===")
print(top_k.to_string(index=False))

# 5. 风险分层
_, risk_summary = segment_customers(y_true, y_prob)

print("\n=== Risk Segmentation ===")
print(risk_summary.to_string(index=False))

# 6. 营销收益模拟
profit = simulate_retention_profit(
    top_k,
    success_rates=(0.1, 0.2, 0.3),
    contact_cost=10,
    retention_value=200,
)

print("\n=== Profit Simulation ===")
print(
    profit.pivot(
        index="fraction",
        columns="success_rate",
        values="expected_net_profit",
    )
)

# 7. 验证核心指标
assert len(df) == 118839
assert abs(metrics["roc_auc"] - 0.9160) < 0.0001
assert abs(metrics["brier_score"] - 0.0954) < 0.0001

print("\nCore validation checks passed!")