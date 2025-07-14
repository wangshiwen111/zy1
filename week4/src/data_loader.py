import pandas as pd
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

def load_data(path: Path) -> pd.DataFrame:
    """读取 CSV 并打印基本信息."""
    df = pd.read_csv(path)
    logging.info(f"数据集形状: {df.shape}")
    logging.info(f"缺失值统计:\n{df.isnull().sum().sort_values(ascending=False)}")
    return df