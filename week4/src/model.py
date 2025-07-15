from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from pathlib import Path
import numpy as np

from .config import RANDOM_STATE, TEST_SIZE, MODEL_PATH, TARGET


def build_pipeline(model_type='rf'):
    """构建不同类型的模型 pipeline"""
    cat = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color', 'Unit of Sale']
    num = ['Month', 'Week']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
        ])

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    elif model_type == 'lgbm':
        model = LGBMRegressor(random_state=RANDOM_STATE)
    elif model_type == 'xgb':
        model = XGBRegressor(random_state=RANDOM_STATE)
    else:
        raise ValueError("Unsupported model type")

    return Pipeline([('preprocessor', preprocessor), ('regressor', model)])


def train_test(df):
    """划分训练/测试集."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def hyperparameter_tuning(pipe, X_train, y_train, model_type):
    """GridSearchCV 调参"""
    param_grid = {}

    if model_type == 'rf':
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2],
            'regressor__max_features': ['auto', 'sqrt']
        }
    elif model_type == 'lgbm':
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [-1, 5, 10],
            'regressor__num_leaves': [20, 31, 50]
        }
    elif model_type == 'xgb':
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__subsample': [0.8, 1.0]
        }

    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def train(df, model_type='rf', tune=False):
    """训练并保存模型"""
    X_train, X_test, y_train, y_test = train_test(df)
    pipe = build_pipeline(model_type)

    # 根据模型类型设置保存路径
    if model_type == 'rf':
        model_path = Path(__file__).resolve().parent.parent / "saved_models" / "best_rf.pkl"
    elif model_type == 'lgbm':
        model_path = Path(__file__).resolve().parent.parent / "saved_models" / "best_lgbm.pkl"
    elif model_type == 'xgb':
        model_path = Path(__file__).resolve().parent.parent / "saved_models" / "best_xgb.pkl"
    else:
        raise ValueError("Unsupported model type")

    if tune:
        pipe = hyperparameter_tuning(pipe, X_train, y_train, model_type)
    else:
        pipe.fit(X_train, y_train)

    joblib.dump(pipe, model_path)
    return pipe, X_test, y_test