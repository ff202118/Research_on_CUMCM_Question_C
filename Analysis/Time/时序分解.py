import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd

seasonal_period = 12
df = pd.read_csv('../../Data_Process/result-sale.csv')
df['日期'] = pd.to_datetime(df['日期'])
grouped = df.groupby('分类名称')
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 时序分解图
plt.figure(figsize=(8, 6))
for name, group in grouped:
    # 进行时序分解
    result = seasonal_decompose(group['销售总量'], model='multiplicative', period=seasonal_period)

    # 绘制时序分解图
    plt.figure(figsize=(12, 6))

    plt.subplot(4, 1, 1)
    plt.plot(group['日期'], group['销售总量'], label=f'原始数据')
    plt.title(f'{name}-时序分解图', fontsize=18)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(group['日期'], result.trend, label=f'趋势成分')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(group['日期'], result.seasonal, label=f'季节成分')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(group['日期'], result.resid, label=f'残差')
    plt.legend()

    plt.tight_layout()

    plt.show()