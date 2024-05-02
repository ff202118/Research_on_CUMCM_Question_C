import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json

df = pd.read_csv('../../Data_Process/result-sale.csv')

df['日期'] = pd.to_datetime(df['日期'])
# df = df[(df['日期'] >= '2023-06-01') & (df['日期'] <= '2023-06-30')]

grouped = df.groupby('分类名称')

cate = ['水生根茎类', '食用菌', '茄类', '辣椒类', '花叶类', '花菜类']
plt.rcParams['font.family'] = 'Microsoft YaHei'

for category, group in grouped:
    print(category)
    x, y = group['ln4单价'], group['ln4总量']
    x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
    detax = 0.01


    nei = np.arange(np.min(x), np.max(x), 0.01)
    # 取中值进行线性回归
    X, Y = [], []
    # for item in x:
    #     Dict = {'value': []}
    #     neighborhood_data = x[(x >= item - detax) & (x <= item + detax)]
    #     neighborhood_indices = np.where((x > item - detax) & (x < item + detax))[0]
    #
    #     for data, idx in zip(neighborhood_data, neighborhood_indices):
    #         Dict['value'].append([data, idx])
    #
    #     sorted(Dict['value'], key=lambda x: x[0])
    #     value_list = Dict['value']
    #     median_value, index = value_list[len(value_list)//2][0], value_list[len(value_list)//2][1]
    #
    #     X.append(median_value)
    #     Y.append(y[index])
    for item in nei:
        neighborhood_data = x[(x >= item - detax) & (x <= item + detax)]
        neighborhood_indices = np.where((x > item - detax) & (x < item + detax))[0]

        if neighborhood_data.size > 0 and neighborhood_indices.size > 0 :
            # print(neighborhood_data)
            # print(y[neighborhood_indices])

            avg_x = np.median(neighborhood_data)
            avg_y = np.median(y[neighborhood_indices])

            # print(avg_x, avg_y)
            X.append(avg_x)
            Y.append(avg_y)

    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)

    # print(X.shape, Y.shape)
    model = LinearRegression()
    model.fit(X, Y)

    intercept = model.intercept_[0]
    slope = model.coef_[0][0]
    sale = np.mean(Y)
    Smax = np.max(Y)

    # 打印拟合参数
    print(f"Ep:{intercept}, d:{slope}")

    with open(f'../LSTM时序预测/save/{category}.json', 'r') as f:
        data = json.load(f)

    data['intercept'] = intercept
    data['slope'] = slope
    data['sale'] = math.exp(sale)
    data['Smax'] = math.exp(Smax)

    with open(f'../LSTM时序预测/save/{category}.json', 'w') as f:
        json.dump(data, f)

    # 绘制散点图和拟合直线
    plt.scatter(X, Y)
    plt.plot(X, model.predict(X), color='red')  # 绘制拟合直线
    plt.title(f'{category}-Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
