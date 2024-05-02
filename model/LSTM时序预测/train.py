import copy
import statistics
import json
import warnings
from matplotlib import pyplot as plt
from utils import create_dataset, split_dataset, load_data
import argparse
from model import Model
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score

LR = 0.01
DP = 0.6
WD = 5e-5
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=LR)
parser.add_argument('--weight_decay', type=float, default=WD)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--dropout', type=float, default=DP)
parser.add_argument('--seq_len', type=int, default=14)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--output_dim', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()


def train(X_train, X_vali, y_train, y_vali):
    hidden = args.hidden
    output_dim = args.output_dim
    batch_size = args.batch_size
    layer_num = args.layer_num

    model = Model(hidden, output_dim, layer_num)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_model = None
    min_vali_loss = 5.0

    for epoch in range(args.epochs):
        train_loss = []

        # Train
        model.train()
        for st in range(0, X_train.shape[0], batch_size):
            seq = X_train[st:st + batch_size]
            label = torch.tensor(y_train[st:st + batch_size].reshape(-1, 1)).float()  # 行数不固定(-1) 列数固定 1 列

            y_pred = model(torch.tensor(seq).float())

            loss = loss_func(y_pred, label)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Vali
        model.eval()
        with torch.no_grad():
            y_vali_pred = model(torch.tensor(X_vali).float())
            loss_vali = loss_func(y_vali_pred, torch.tensor(y_vali.reshape(-1, 1)).float())

        if loss_vali < min_vali_loss:
            min_vali_loss = loss_vali
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), loss_vali))

    state = {'models': best_model.state_dict()}
    torch.save(state, path)


def test(X, y):
    # Load model
    print('loading model...')

    hidden = args.hidden
    output_dim = args.output_dim
    layer_num = args.layer_num

    model = Model(hidden, output_dim, layer_num)
    model.load_state_dict(torch.load(path)['models'])

    # Test
    print('Testing..')
    model.eval()
    pred = []
    y_true = []

    for (seq, label) in tqdm(zip(X, y)):
        seq = seq.reshape(1, args.seq_len + 1)
        y_true.append(label)
        with torch.no_grad():
            y_pred = model(torch.tensor(seq).float())
            pred.append(y_pred.numpy())

    y_true, pred = np.array(y_true), np.array(pred).reshape(len(y_true))

    print(y_true[:2])
    print(pred[:2])
    # 计算R方
    r_squared = r2_score(y_true, pred)
    print(f'R-squared: {r_squared}')

    percentage_errors = np.abs((y_true - pred) / y_true) * 100
    MAPE = np.mean(percentage_errors)
    print(f"MAPE: {MAPE}")

    squared_errors = (y_true - pred) ** 2
    RMSE = np.sqrt(np.mean(squared_errors))
    print(f"RMSE: {RMSE}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pred, color='red', label='Prediction')
    plt.plot(y_true, color='blue', label='Truth')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend(loc='best')
    plt.show()
    return r_squared, MAPE, RMSE


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    plt.rcParams['font.family'] = 'Microsoft YaHei'

    for cate_id in range(0, 6):
        cate = ['茄类', '辣椒类', '花菜类', '花叶类', '水生根茎类', '食用菌']
        group = load_data(cate_id)
        path = f'最优模型/{cate[cate_id]}.pth'
        SEQ_LEN = args.seq_len
        # X: [batch_size, seq_len], Y: [batch_size, 1]
        X, y = create_dataset(group, seq_len=SEQ_LEN)

        X_train, X_vali, X_test, y_train, y_vali, y_test = split_dataset(X, y, train_rate=0.8)

        train(X_train, X_vali, y_train, y_vali)

        R2, MAPE, RMSE = test(X_test, y_test)

        # 将信息组织成一个字典

        info_dict = {
            'R2': R2,
            'MAPE': MAPE,
            'RMSE': RMSE,
        }

        # 将字典写入 JSON 文件
        json_file_path = f'./save/{cate[cate_id]}.json'
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        data['R2'], data['MAPE'], data['RMSE'] = R2, MAPE, RMSE

        with open(json_file_path, 'w') as f:
            json.dump(data, f)
