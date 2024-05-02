import matplotlib.pyplot as plt
import json
import numpy as np
import random
import math


def objective_function(C, P, S, R):
    Sum = 0.0
    for j in range(len(C)):
        Sum += S[j] * (P[j] - C[j]) - R[j] * C[j]
    return -Sum


# 定义接受新解的条件
def accept_condition(delta, temperature):
    if delta < 0:
        return True
    else:
        return random.random() < math.exp(-delta / temperature)


# 定义邻域搜索函数
def neighbor_search(Sale, current_S, current_P, current_R, Ep, C, gamma, Smax, step_size):
    new_S = [s for s in current_S]
    new_P = [p for p in current_P]
    new_R = [r for r in current_R]
    i = random.randint(0, len(new_S) - 1)

    new_P[i] = max(1.1 * C[i], min(1.5 * C[i], new_P[i] + random.uniform(-step_size, step_size)))
    new_S[i] = math.exp(Sale + Ep * math.log(new_P[i]))
    new_R[i] = min(Smax, new_S[i] * (1 + gamma))

    return new_S, new_P, new_R


# 模拟退火函数
def simulated_annealing(Sale, C, gamma, Smax, initial_S, Ep, initial_P, initial_R,
                        initial_temperature, cooling_rate, stopping_temperature, step_size):
    Max_solution = 0
    Best_S = None
    Best_R = None
    Best_P = None

    current_S = initial_S
    current_P = initial_P
    current_R = initial_R
    current_solution = objective_function(C, current_P, current_S, current_R)
    current_temperature = initial_temperature

    best_solution_at_temperature = []  # 记录每个温度下的最佳解
    temperatures = []  # 记录温度

    while current_temperature > stopping_temperature:
        new_S, new_P, new_R = neighbor_search(Sale, current_S, current_P, current_R, Ep, C, gamma, Smax, step_size)
        new_solution = objective_function(C, new_P, new_S, new_R)
        delta = new_solution - current_solution

        if accept_condition(delta, current_temperature):
            current_S = new_S
            current_P = new_P
            current_R = new_R
            current_solution = new_solution

        # 记录每个温度下的最佳解
        best_solution_at_temperature.append(current_solution)
        temperatures.append(current_temperature)

        # 降温
        current_temperature *= cooling_rate

        if current_solution > Max_solution:
            Max_solution = current_solution
            Best_P = current_P
            Best_R = current_R
            Best_S = current_S

    return Best_S, Best_P, Best_R, best_solution_at_temperature, temperatures

if __name__ == '__main__':
    cate = ['水生根茎类', '食用菌', '茄类', '辣椒类', '花叶类', '花菜类']
    plt.rcParams['font.family'] = 'Microsoft YaHei'

    for category in cate:
        with open(f'../LSTM时序预测/save/{category}.json', 'r') as f:
            data = json.load(f)

        Std, Mean, Gamma, D, Ep, Sale, C, Smax = data['std'], data['mean'], \
          data['r'], data['intercept'], data['slope'], data['sale'], data['pred'], data['Smax']

        # 初始参数
        initial_S = [0 for _ in range(7)]  # 初始销量
        initial_P = [random.uniform(1.1 * C[j], 1.5 * C[j]) for j in range(7)]  # 初始定价
        initial_R = [0 for _ in range(7)]  # 初始补货量
        initial_temperature = 1000
        cooling_rate = 0.99
        stopping_temperature = 0.01
        step_size = 1.0

        # 使用模拟退火进行优化
        Best_S, Best_P, Best_R, best_solution_at_temperature, temperatures = simulated_annealing(
            D, C, Gamma, Smax, initial_S, Ep, initial_P, initial_R,
            initial_temperature, cooling_rate, stopping_temperature, step_size)


        print(category)
        print("Best S(销量):", Best_S)
        print("Best P(定价):", Best_P)
        print("Best R(补货量):", Best_R)

        # 绘制温度和目标函数值之间的关系
        plt.plot(temperatures, best_solution_at_temperature, label=category)

    plt.xlabel('温度')
    plt.ylabel('目标函数值')
    plt.title('模拟退火过程中的目标函数值变化')
    plt.legend()
    plt.show()
