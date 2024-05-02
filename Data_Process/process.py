import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.family'] = 'Microsoft YaHei'


# 主函数
path = './result.csv'  # 数据路径
# 提取数据
dataset = pd.read_csv(path)
# 将数据按分类拆分
grouped = dataset.groupby('蔬菜品类')


def get_data(category, index):
    group = grouped.get_group(category).reset_index()

    # 格式处理
    group['日期'] = pd.to_datetime(group['日期'])
    group['批发价格'] = pd.to_numeric(group['批发价格'])
    data = group['批发价格'].to_numpy().reshape(-1, 1)

    # 异常值处理
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    std = np.std(data)
    mean = np.mean(data)

    r = [11.97473684, 8.130972222, 7.122, 8.515333333, 10.2803, 14.142]
    info_dict = {
        'category': category,
        'std': std,
        'mean': mean,
        'r': 0.01 * r[index]
    }


    for index, item in enumerate(data_std):
        if abs(item) > 3.0:
            group.loc[index, '批发价格'] = np.mean(data[index:index+7])
        else:
            group.loc[index, '批发价格'] = item * std + mean


    plt.figure(figsize=(8, 4.5))
    plt.plot(group['日期'], group.loc[:, '批发价格'])
    plt.title(f'{category}')
    plt.xlabel('日期/天')
    plt.ylabel('批发价格(元/kg)')
    plt.savefig(f'./cate_data/image/{category}.png')
    plt.show()

    df = pd.concat([group['日期'], group['批发价格']], axis=1)
    df.columns = ['日期', '批发价格']
    df.to_csv(f'./cate_data/result-{category}.csv')

    json_file_path = f'../model/LSTM时序预测/save/{category}.json'
    with open(json_file_path, 'w') as f:
        json.dump(info_dict, f)

if __name__ == '__main__':
    cate = ['水生根茎类', '食用菌', '茄类', '辣椒类', '花叶类', '花菜类']
    for index in range(0, 6):
        category = cate[index]
        get_data(category, index)