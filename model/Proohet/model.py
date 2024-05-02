from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from utils import load_data
import matplotlib.pyplot as plt
import json

for cate_id in range(0, 6):
    cate = ['茄类', '辣椒类', '花菜类', '花叶类', '水生根茎类', '食用菌']
    group = load_data(cate_id, model='prophet')
    group['日期'] = pd.to_datetime(group['日期'])

    group = group.rename(columns={'日期': 'ds', '批发价格': 'y'})

    df_train = group[(group['ds'] >= '2020-07-01') & (group['ds'] <= '2023-6-30')]
    Time_len = len(df_train)
    # df_test = group[(group['ds'] >= '2023-06-01') & (group['ds'] <= '2023-6-30')]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.add_country_holidays(country_name="CN")
    model.fit(df_train)

    future = model.make_future_dataframe(periods=7, freq='D')
    forecast = model.predict(future)


    model.plot(forecast)
    plt.show()

    model.plot_components(forecast)


    # 计算R方
    def compute_r_squared(y_true, y_pred):
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_total)
        return r_squared

    y_true = np.array(group['y'])
    y = np.array(forecast['yhat'])

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, color='red', label='Truth')
    plt.plot(y, color='blue', label='Prediction')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend(loc='best')
    plt.show()

    R2 = compute_r_squared(group['y'], forecast['yhat'][:Time_len])
    print("R方值：", R2)

    percentage_errors = np.abs((group['y'] - forecast['yhat'][:Time_len]) / group['y']) * 100
    MAPE = np.mean(percentage_errors)
    print("平均绝对百分比误差：", MAPE)


    mse = mean_absolute_error(group['y'], forecast['yhat'][:Time_len])
    RMSE = np.sqrt(mse)
    print("均方根误差（RMSE）：", RMSE)


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

    values = []
    for item in forecast['yhat'][Time_len+1:]:
        values.append(item)

    data['pred'] = values

    with open(json_file_path, 'w') as f:
        json.dump(data, f)
