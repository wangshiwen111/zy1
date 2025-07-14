import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def price_distribution(df):
    price_cols = ['Low Price', 'High Price', 'Mostly Low', 'Mostly High']
    df[price_cols].hist(bins=30, figsize=(12, 8))
    plt.suptitle('价格分布直方图')
    plt.show()

def key_features_dist(df):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    sns.countplot(data=df, x='Variety', ax=ax[0][0], order=df['Variety'].value_counts().iloc[:5].index)
    ax[0][0].tick_params(axis='x', rotation=45)
    sns.countplot(data=df, x='Origin', ax=ax[0][1], order=df['Origin'].value_counts().iloc[:5].index)
    ax[0][1].tick_params(axis='x', rotation=45)
    sns.countplot(data=df, x='Item Size', ax=ax[1][0])
    ax[1][0].tick_params(axis='x', rotation=45)
    sns.scatterplot(data=df, x='Low Price', y='High Price', hue='Variety', alpha=0.6, ax=ax[1][1])
    plt.tight_layout()
    plt.show()

def monthly_box(df):
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Month', y='Low Price')
    plt.title('按月分析南瓜价格')
    plt.show()