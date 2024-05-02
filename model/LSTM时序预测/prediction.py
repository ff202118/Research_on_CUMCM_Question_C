import json

import torch
from model import Model
from train import args
from utils import load_data, create_dataset
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'

def pred(path, X, date=7):
    hidden = args.hidden
    output_dim = args.output_dim
    layer_num = args.layer_num
    model = Model(hidden, output_dim, layer_num)
    model.load_state_dict(torch.load(path)['models'])

    preds = []

    with torch.no_grad():
        for i in range(date):
            X = X.reshape(1, SEQ_LEN + 1)
            y_pre = model(torch.tensor(X).float()).numpy()

            preds.append(y_pre[0, 0])

            X = np.append(X[:, 1:], y_pre)

        return preds


if __name__ == '__main__':

    for cate_id in range(0, 6):
        cate = ['茄类', '辣椒类', '花菜类', '花叶类', '水生根茎类', '食用菌']
        path = f'最优模型/{cate[cate_id]}.pth'
        SEQ_LEN = args.seq_len
        group = load_data(cate_id)

        X = np.array(group.loc[len(group) - SEQ_LEN - 1:, '批发价格'])
        preds = pred(path, X, date=7)

        json_file_path = f'./save/{cate[cate_id]}.json'
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        std = data['std']
        mean = data['mean']

        before = np.array(group.loc[len(group) - SEQ_LEN - 1:, '批发价格'])

        plt.plot(range(0, len(before)), before * std + mean)
        plt.plot(range(len(before), len(before)+len(preds)), np.array(preds) * std + mean)
        plt.title(f'{cate[cate_id]}-Linear Regression')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        values = []
        for item in preds:
            values.append(item * std + mean)

        data['pred'] = values

        with open(json_file_path, 'w') as f:
            json.dump(data, f)