from src.data import load_data, prepare_features, encode_categories

# 1. 读取数据
train, test = load_data("data/train.csv", "data/test.csv")

# 2. 划分特征与目标变量
X, y, X_test = prepare_features(train, test)

# 3. 统一类别变量定义
X, X_test, cat_cols = encode_categories(X, X_test)

# 4. 检查结果
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)
print("Categorical features:", cat_cols)

assert X.shape[1] == X_test.shape[1] == 19
assert X.columns.equals(X_test.columns)
assert set(y.unique()) == {0, 1}
assert all(str(X[col].dtype) == "category" for col in cat_cols)
assert all(
    X[col].cat.categories.equals(X_test[col].cat.categories)
    for col in cat_cols
)

print("All checks passed!")

