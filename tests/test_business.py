from src.business import (
    segment_customers,
    simulate_retention_profit,
)
from src.evaluation import evaluate_top_k

# 构造简单测试数据
y_true = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.4, 0.6, 0.15, 0.95]

# 1. 测试风险分层
df, summary = segment_customers(y_true, y_prob)

print("Risk segmentation:")
print(summary)

assert len(summary) == 5
assert summary["customers"].sum() == 10
assert summary["actual_churn"].sum() == 5

# 2. 测试营销收益模拟
top_k = evaluate_top_k(
    y_true, y_prob, fractions=(0.2, 0.4)
)

profit = simulate_retention_profit(
    top_k,
    success_rates=(0.2,),
    contact_cost=10,
    retention_value=200,
)

print("\nProfit simulation:")
print(profit)

assert len(profit) == 2

# Top 20%: contact 2 customers, capture 2 churners
# Expected profit = 2 * 0.2 * 200 - 2 * 10 = 60
assert profit.iloc[0]["expected_net_profit"] == 60

print("\nAll business tests passed!")
