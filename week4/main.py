from src.utils import price_distribution, key_features_dist, monthly_box
from src.data_loader import load_data
from src.preprocess import full_pipeline
from src.model import train
from src.evaluate import evaluate, feature_importance
from src.config import DATA_PATH

if __name__ == "__main__":
    raw_df = load_data(DATA_PATH)      # 原始数据
    price_distribution(raw_df)         # 原始列还在
    key_features_dist(raw_df)
    monthly_box(raw_df)

    df = full_pipeline(raw_df)         # 再做清洗
    model, X_test, y_test = train(df, tune=False)
    evaluate(model, X_test, y_test)
    feature_importance(model)