from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "US-pumpkins.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "best_rf.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Avg_Price"