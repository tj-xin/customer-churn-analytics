
from src.evaluation import evaluate_classifier, evaluate_top_k

y_true = [0, 1, 0, 1, 1]
y_prob = [0.1, 0.9, 0.2, 0.8, 0.7]

metrics = evaluate_classifier(y_true, y_prob)
top_k = evaluate_top_k(y_true, y_prob, fractions=(0.2, 0.4))

print("ROC-AUC:", metrics["roc_auc"])
print("PR-AUC:", metrics["pr_auc"])
print(top_k)

assert metrics["roc_auc"] == 1.0
assert metrics["pr_auc"] == 1.0
assert top_k.iloc[0]["captured_churn"] == 1
assert top_k.iloc[1]["captured_churn"] == 2

print("All evaluation tests passed!")
