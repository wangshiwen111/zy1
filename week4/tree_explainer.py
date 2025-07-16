# tree_explainer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from src.data_loader import load_data
from src.preprocess import full_pipeline
from src.model import train
from src.config import DATA_PATH

# 1. 数据 & 随机森林
df_raw = load_data(DATA_PATH)
df = full_pipeline(df_raw)
model, _, _ = train(df, model_type='rf', tune=False)

# 2. 特征名称（兼容 sklearn <1.3 与 ≥1.3）
pre = model.named_steps['preprocessor']
cat_cols = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color', 'Unit of Sale']
num_cols = ['Month', 'Week']

# sklearn <1.3 用 get_feature_names，≥1.3 用 get_feature_names_out
try:
    cat_names = pre.named_transformers_['cat'].get_feature_names_out(cat_cols)
except AttributeError:          # 旧版本 sklearn
    cat_names = pre.named_transformers_['cat'].get_feature_names(cat_cols)

feature_names = list(cat_names) + num_cols

# 3. 随机取一棵树
tree = model.named_steps['regressor'].estimators_[0]

# 4. 可视化（3 层即可）
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, max_depth=3)
plt.title("Random Forest – Single Tree (depth=3)")
plt.show()

# 5. 随机一条根→叶路径
left  = tree.children_left
right = tree.children_right
feat  = tree.feature
thres = tree.threshold
value = tree.value

node = 0
path = [node]
import random
while left[node] != right[node]:  # 非叶子
    name = feature_names[feat[node]]
    direction = random.choice(['left', 'right'])
    if direction == 'left':
        node = left[node]
        print(f"Split: {name} <= {thres[node]:.2f}")
    else:
        node = right[node]
        print(f"Split: {name} > {thres[node]:.2f}")
    path.append(node)

print("Path:", path, "→ 预测值:", value[node][0])