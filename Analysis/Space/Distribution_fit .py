import numpy as np
from scipy.stats import lognorm, kstest
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('../../Data_Process/result-sale.csv')
df['日期'] = pd.to_datetime(df['日期'])
grouped = df.groupby('分类名称')
plt.rcParams['font.family'] = 'Microsoft YaHei'

cate = []
data_sets = []
parameters = []
X = []

for category, group in grouped:
    print(category)
    data = group['销售总量']
    data = np.array(data)

    data_log = np.log(data)
    data_sets.append(data_log)

    shape, loc, scale = lognorm.fit(data_log)
    parameters.append((shape, loc, scale))

    x = np.linspace(min(data_log), max(data_log), 1000)
    X.append(x)
    cate.append(category)


    # 进行K-S检验
    _, p_value = kstest(data, 'lognorm', args=(shape, loc, scale))

    # 打印结果
    print("参数 shape: ", shape)
    print("参数 loc: ", loc)
    print("参数 scale: ", scale)
    print("K-S检验p值: ", p_value)


plt.figure(figsize=(10, 6))

for i, data in enumerate(data_sets):
    shape, loc, scale = parameters[i]
    # 绘制直方图
    plt.hist(data, bins=30, density=True, alpha=0.4, color='#518cba')
    x = X[i]
    # 绘制拟合曲线
    plt.plot(x, lognorm.pdf(x, shape, loc, scale), label=f'{cate[i]}')



plt.xlabel('对数值')
plt.ylabel('概率密度')
plt.title('对数正态分布拟合')
plt.legend()
plt.grid(True)
plt.show()
