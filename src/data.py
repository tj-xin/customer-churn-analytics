import pandas as pd


def load_data(train_path, test_path=None):
    """Load Kaggle training and optional test data."""
    train = pd.read_csv(train_path)

    if test_path is None:
        return train

    test = pd.read_csv(test_path)
    return train, test


def prepare_features(train, test=None):
    """Prepare features and binary churn labels."""
    X = train.drop(columns=["id", "Churn"]).copy()
    y = (train["Churn"] == "Yes").astype(int)

    if test is None:
        return X, y

    X_test = test[X.columns].copy()
    return X, y, X_test



def encode_categories(X_train, X_other=None):
    """Align categorical feature definitions across datasets."""
    X_train = X_train.copy()

    cat_cols = X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    if "SeniorCitizen" in X_train.columns:
        cat_cols.append("SeniorCitizen")

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")

    if X_other is None:
        return X_train, cat_cols

    X_other = X_other.copy()

    for col in cat_cols:
        X_other[col] = pd.Categorical(
            X_other[col],
            categories=X_train[col].cat.categories
        )

    return X_train, X_other, cat_cols
