import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(index):
    cate = ['茄类', '辣椒类', '花菜类', '花叶类', '水生根茎类', '食用菌']
    category = cate[index]
    group = pd.read_csv(f'../../Data_Process/cate_data/result-{category}.csv')

    # 数据标准化
    scaler = StandardScaler()
    group['批发价格'] = scaler.fit_transform(group['批发价格'].to_numpy().reshape(-1, 1))

    return group

def create_dataset(group, seq_len=6):
    X = []
    y = []

    start = 0
    end = group.shape[0] - seq_len

    for i in range(start, end):
        sample = group.loc[i: i + seq_len, '批发价格']  # 基于时间跨度seq_len创建样本
        label = group.loc[i + seq_len, '批发价格']  # 创建sample对应的标签
        X.append(sample)
        y.append(label)

    return np.array(X), np.array(y)

def split_dataset(X, y, train_rate=0.7, vali_rate=0.1):
    X_len = len(X)
    train_data_len = int(X_len * train_rate)
    vali_data_len = int(X_len * vali_rate)

    X_train = X[:train_data_len]
    y_train = y[:train_data_len]

    X_vali = X[train_data_len:train_data_len+vali_data_len]
    y_vali = y[train_data_len:train_data_len+vali_data_len]

    X_test = X[train_data_len+vali_data_len:]
    y_test = y[train_data_len+vali_data_len:]

    # 返回值
    return X_train, X_vali, X_test, y_train, y_vali, y_test