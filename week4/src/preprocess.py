import pandas as pd
import numpy as np

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """创建目标变量."""
    df["Avg_Price"] = (df["Mostly Low"] + df["Mostly High"]) / 2
    return df

def select_features(df: pd.DataFrame, keep: list) -> pd.DataFrame:
    """保留指定列."""
    return df[keep + ["Date", "Avg_Price"]].copy()

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """日期特征工程."""
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype("int64")
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值填充."""
    df["Color"].fillna("UNKNOWN", inplace=True)
    df["Item Size"].fillna("MEDIUM", inplace=True)
    df["Unit of Sale"].fillna("BIN", inplace=True)
    return df

def remove_outliers(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """基于 IQR 去除异常值."""
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[target] < (Q1 - 1.5 * IQR)) | (df[target] > (Q3 + 1.5 * IQR)))
    return df[mask].copy()

def full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """整合所有预处理步骤."""
    df = create_target(df)
    features = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color', 'Unit of Sale']
    df = select_features(df, features)
    df = feature_engineering(df)
    df = handle_missing(df)
    df = remove_outliers(df, "Avg_Price")
    return df.dropna(subset=["Avg_Price"])