import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


def evaluate(model, X_test, y_test):
    """评估指标 + 画图."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.3f}, R2: {r2:.3f}")

    # 实际 vs 预测
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('实际价格');
    plt.ylabel('预测价格');
    plt.title('实际 vs 预测')
    plt.show()

    # 残差图
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=.6)
    plt.axhline(0, color='r', ls='--')
    plt.xlabel('预测价格');
    plt.ylabel('残差');
    plt.title('残差图')
    plt.show()

    return {"mse": mse, "r2": r2}


def feature_importance(model):
    """可视化前 15 重要特征."""
    importances = None

    if isinstance(model.named_steps['regressor'], (RandomForestRegressor, LGBMRegressor)):
        importances = model.named_steps['regressor'].feature_importances_
    elif isinstance(model.named_steps['regressor'], XGBRegressor):
        importances = model.named_steps['regressor'].feature_importances_
    else:
        raise ValueError("Unsupported model type for feature importance")

    cat = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color', 'Unit of Sale']
    num = ['Month', 'Week']
    cat_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat)
    feature_names = np.concatenate([cat_names, num])
    sorted_idx = importances.argsort()[-15:][::-1]  # 降序排列

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names[sorted_idx], importances[sorted_idx])
    plt.title("Top 15 特征重要性")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()