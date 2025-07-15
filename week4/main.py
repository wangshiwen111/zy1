from src.data_loader import load_data
from src.preprocess import full_pipeline
from src.model import train
from src.evaluate import evaluate, feature_importance
from src.utils import price_distribution, key_features_dist, monthly_box
from src.config import DATA_PATH

if __name__ == "__main__":
    df_raw = load_data(DATA_PATH)          # 1. 先拿到原始数据
    price_distribution(df_raw)             # 2. 用原始数据画图
    key_features_dist(df_raw)
    monthly_box(df_raw)

    df = full_pipeline(df_raw)             # 3. 再做清洗、特征工程
    for model_type in ['rf', 'lgbm', 'xgb']:
        print(f"Training {model_type.upper()} model...")
        model, X_test, y_test = train(df, model_type=model_type, tune=False)
        evaluate(model, X_test, y_test)
        feature_importance(model)

import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.model import build_pipeline
from src.config import RANDOM_STATE, TARGET

def generate_json_output(df, model_type='rf'):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fold_results = {
        "model_name": model_type.upper(),
        "model_params": None,
        "fea_encoding": "one-hot"
    }

    train_rmse_list, train_mae_list, train_r2_list = [], [], []
    test_rmse_list, test_mae_list, test_r2_list = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = build_pipeline(model_type)
        pipe.fit(X_train, y_train)

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5

        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        fold_results[f"{fold}_fold_train_data"] = [len(X_train), X_train.shape[1]]
        fold_results[f"{fold}_fold_test_data"] = [len(X_test), X_test.shape[1]]
        fold_results[f"{fold}_fold_train_performance"] = {
            "rmse": f"{train_rmse:.2f}",
            "mae": f"{train_mae:.2f}",
            "r2": f"{train_r2:.2f}"
        }
        fold_results[f"{fold}_fold_test_performance"] = {
            "rmse": f"{test_rmse:.2f}",
            "mae": f"{test_mae:.2f}",
            "r2": f"{test_r2:.2f}"
        }

        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        train_r2_list.append(train_r2)
        test_rmse_list.append(test_rmse)
        test_mae_list.append(test_mae)
        test_r2_list.append(test_r2)

    fold_results["average_train_performance"] = {
        "rmse": f"{np.mean(train_rmse_list):.2f}",
        "mae": f"{np.mean(train_mae_list):.2f}",
        "r2": f"{np.mean(train_r2_list):.2f}"
    }
    fold_results["average_test_performance"] = {
        "rmse": f"{np.mean(test_rmse_list):.2f}",
        "mae": f"{np.mean(test_mae_list):.2f}",
        "r2": f"{np.mean(test_r2_list):.2f}"
    }

    return fold_results
output = []
for mt in ['rf', 'lgbm', 'xgb']:
    print(f"Generating JSON for {mt.upper()}...")
    output.append(generate_json_output(df, model_type=mt))

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

# 单轮实验：LGBM 与 XGBoost
import json
from src.evaluate import evaluate
from src.model import train

results = []
for mt in ['lgbm', 'xgb']:
    print(f"【单轮实验】{mt.upper()} 训练开始...")
    model, X_test, y_test = train(df, model_type=mt, tune=False)
    metrics = evaluate(model, X_test, y_test)
    results.append({
        "model": mt.upper(),
        "MSE": round(metrics["mse"], 3),
        "R2": round(metrics["r2"], 3)
    })

# 保存实验结果
with open("experiment_single.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)