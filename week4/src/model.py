from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from .config import RANDOM_STATE, TEST_SIZE, MODEL_PATH, TARGET

def build_pipeline() -> Pipeline:
    """构建默认 pipeline."""
    cat = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color', 'Unit of Sale']
    num = ['Month', 'Week']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
        ])
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    return Pipeline([('preprocessor', preprocessor), ('regressor', model)])

def train_test(df):
    """划分训练/测试集."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def hyperparameter_tuning(pipe, X_train, y_train):
    """GridSearchCV 调参."""
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__max_features': ['auto', 'sqrt']
    }
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)
    return gs.best_estimator_

def train(df, tune=False):
    """训练并保存模型."""
    X_train, X_test, y_train, y_test = train_test(df)
    pipe = build_pipeline()
    if tune:
        pipe = hyperparameter_tuning(pipe, X_train, y_train)
    else:
        pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    return pipe, X_test, y_test