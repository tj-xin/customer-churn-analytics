
import numpy as np
import pandas as pd


def segment_customers(y_true, y_prob, n_groups=5):
    """Divide customers into equal-sized churn risk groups."""
    df = pd.DataFrame({
        "Churn": np.asarray(y_true),
        "ChurnProbability": np.asarray(y_prob),
    })

    if len(df) == 0 or len(df) != len(y_true):
        raise ValueError("Input arrays must be non-empty and equal in length.")

    if n_groups < 2 or n_groups > len(df):
        raise ValueError("Invalid number of risk groups.")

    if not df["Churn"].isin([0, 1]).all():
        raise ValueError("Churn labels must be binary.")

    if not np.isfinite(df["ChurnProbability"]).all():
        raise ValueError("Probabilities must be finite.")

    if not df["ChurnProbability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    labels = [
        "Very Low", "Low", "Medium", "High", "Very High"
    ] if n_groups == 5 else [
        f"Group {i + 1}" for i in range(n_groups)
    ]

    df["RiskGroup"] = pd.qcut(
        df["ChurnProbability"].rank(method="first"),
        q=n_groups,
        labels=labels,
    )

    summary = (
        df.groupby("RiskGroup", observed=True)
        .agg(
            customers=("Churn", "size"),
            actual_churn=("Churn", "sum"),
            churn_rate=("Churn", "mean"),
            avg_predicted_risk=("ChurnProbability", "mean"),
        )
        .reset_index()
    )

    return df, summary


def simulate_retention_profit(
    top_k_results,
    success_rates=(0.1, 0.2, 0.3),
    contact_cost=10,
    retention_value=200,
):
    """Simulate hypothetical retention campaign net profits."""
    if contact_cost < 0 or retention_value < 0:
        raise ValueError("Costs and values must be non-negative.")

    rows = []

    for _, result in top_k_results.iterrows():
        k = int(result["contacted"])
        captured = int(result["captured_churn"])

        for success_rate in success_rates:
            if not 0 <= success_rate <= 1:
                raise ValueError("Success rates must be between 0 and 1.")

            expected_retained = captured * success_rate
            expected_profit = (
                expected_retained * retention_value
                - k * contact_cost
            )

            rows.append({
                "fraction": result["fraction"],
                "contacted": k,
                "captured_churn": captured,
                "success_rate": success_rate,
                "expected_retained": expected_retained,
                "expected_net_profit": expected_profit,
            })

    return pd.DataFrame(rows)
