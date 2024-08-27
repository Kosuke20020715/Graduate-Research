import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# hiv数理モデル
def hiv_model(X, V, Y, Z, action):
    lambda_ = 10
    mu_v = 2.4
    mu_y = 0.02
    mu_x = 0.02
    mu_z = 0.24
    r = 0.03
    P_max = 1500
    k1 = 2.4e-05
    k2 = 0.0030
    N = 1200

    dX = lambda_ / (1 + V) - mu_x * X + r * X * (1 - (X + Y + Z) / P_max) - k1 * X * V
    dY = k1 * X * V - mu_y * Y - k2 * Y
    dZ = k2 * Y - mu_z * Z
    dV = (1 - action) * N * mu_z * Z - k1 * X * V - mu_v * V

    dt = 0.15 # 刻み幅
    X_new = X + dX * dt
    V_new = V + dV * dt
    Y_new = Y + dY * dt
    Z_new = Z + dZ * dt

    observation = [X_new, V_new, Y_new, Z_new]
    return observation

# 状態空間と行動空間の定義
X_min, X_max, X_interval = 600, 3000, 10
V_min, V_max, V_interval = 0, 5, 0.05
actions = np.arange(0, 1.1, 0.1)
num_actions = len(actions)
q_table = np.random.uniform(low=-1, high=1, size=(240 * 100, num_actions))

# binsの定義
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 状態を離散化して一意の整数に変換する関数
def digitize_state(X, V):
    digitized = [
        np.digitize(X, bins=bins(X_min, X_max, 240)),
        np.digitize(V, bins=bins(V_min, V_max, 100))
    ]
    digital_num = (digitized[0]+1)*(digitized[1]+1)
    return digital_num

# Q値の更新式
def update_Qtable(state, action, reward, state_next, gamma):
    alpha = 0.1
    Max_Q_next = max(q_table[state_next][:])
    q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * Max_Q_next - q_table[state, action])
    return q_table[state, action]

# 行動の定義
def decide_action(state, tau):
    a_max = max(q_table[state][:])  # 最大値を計算
    sum_exp_values = sum([np.exp((v - a_max) / tau) for v in q_table[state][:]])
    p = [np.exp((v - a_max) / tau) / sum_exp_values for v in q_table[state][:]]
    # ここでpはソフトマックス関数の出力
    # print(q_table[state][:])
    if np.any(np.isnan(p)):
        print(f"Warning: NaN detected in probabilities for state {state}")
    action_index = np.random.choice(np.arange(num_actions), p=p)
    action = actions[action_index]
    return action, action_index

# Q学習アルゴリズム
def q_learning(X, V, Y, Z):
    X_vals, V_vals, Y_vals, Z_vals, action_list = [], [], [], [], []
    for episode in range(NUM_EPISODES):
        if episode == 0:
            X_vals.append(X)
            V_vals.append(V)
            Y_vals.append(Y)
            Z_vals.append(Z)

            state = digitize_state(X, V)
            tau = 1.0
            action, action_index = decide_action(state, tau)
            action_list.append(action)

            # tau = tau * 0.999
            # 温度係数の更新(指数ver)
            T_0 = 1.0
            k=0.1
            tau = T_0 * np.exp(-k * episode)

            observation_next = hiv_model(X, V, Y, Z, action)
            X_new, V_new = observation_next[0:2]

            state_next = digitize_state(X_new, V_new)
            prev_action = 0        
            
             # 報酬の計算(追記)
            # print(np.log(action+1e-8))
            reward = np.log(V / V_new) - (action - prev_action) * np.log(action+1e-12)

            prev_action = action
            q_table[state, action_index] = update_Qtable(state, action_index, reward, state_next, 0.5)

            state = state_next
            X, V = X_new, V_new
        
        else:
            action, action_index = decide_action(state, tau)
            action_list.append(action)
            # tau = tau * 0.999
            # 温度係数の更新(指数ver)

            T_0 = 1.0
            k=0.1
            tau = T_0 * np.exp(-k * episode)

            observation_next = hiv_model(X, V, Y, Z, action)
            X_new, V_new, Y, Z = observation_next

            state_next = digitize_state(X_new, V_new)

            # print(np.log(action+1e-8))
            #報酬の計算
            reward =  np.log(V / V_new) - (action - prev_action) * np.log(action+1e-12)

            prev_action = action

            q_table[state, action_index] = update_Qtable(state, action_index, reward, state_next, 0)

            state = state_next
            X, V = X_new, V_new

            X_vals.append(X)
            V_vals.append(V)
            Y_vals.append(Y)
            Z_vals.append(Z)

    return X_vals, V_vals, Y_vals, Z_vals, action_list

NUM_EPISODES = 500

# 初期条件の分布から200個のサンプルを生成
X_mean, X_std = 1000, 200
Y_mean, Y_std = 2, 1
Z_mean, Z_std = 0.01, 0.01
V_mean, V_std = 1, 1

X_initials = np.clip(np.random.normal(X_mean, X_std, 200), 600, 3000)#600-3000
Y_initials = np.clip(np.random.normal(Y_mean, Y_std, 200), 0, None)
Z_initials = np.clip(np.random.normal(Z_mean, Z_std, 200), 0, None)
V_initials = np.clip(np.random.normal(V_mean, V_std, 200), 1e-12, 5)#0.05-5

X_vals_all, V_vals_all, Y_vals_all, Z_vals_all = [], [], [], []

# 200個の初期条件でシミュレーションを実行
for X, Y, Z, V in zip(X_initials, Y_initials, Z_initials, V_initials):
    X_vals, V_vals, Y_vals, Z_vals, action_list = q_learning(X, V, Y, Z)
    X_vals_all.append(X_vals)
    V_vals_all.append(V_vals)
    Y_vals_all.append(Y_vals)
    Z_vals_all.append(Z_vals)

#サンプルパスの抽出
sample_X_vals = random.sample(X_vals_all,30)
sample_Y_vals = random.sample(Y_vals_all,30)
sample_Z_vals = random.sample(Z_vals_all,30)
sample_V_vals = random.sample(V_vals_all,30)


# グラフの表示
plt.figure(figsize=(10, 10))

# Xのプロット
plt.subplot(2, 2, 1)
for X_vals in sample_X_vals:
    plt.plot(X_vals, color='blue',)
plt.xlabel('days')
plt.ylabel('X')
plt.title('X over 500 Episodes')

# Vのプロット
plt.subplot(2, 2, 2)
for V_vals in sample_V_vals:
    plt.plot(V_vals, color='orange',)
plt.xlabel('days')
plt.ylabel('V')
plt.title('V over 500 Episodes')

# Yのプロット
plt.subplot(2, 2, 3)
for Y_vals in sample_Y_vals:
    plt.plot(Y_vals, color='red', )
plt.xlabel('days')
plt.ylabel('Y')
plt.title('Y over 500 Episodes')

# Zのプロット
plt.subplot(2, 2, 4)
for Z_vals in sample_Z_vals:
    plt.plot(Z_vals, color='green', )
plt.xlabel('days')
plt.ylabel('Z')
plt.title('Z over 500 Episodes')

plt.tight_layout()
plt.show()


# データをcsvに保存

data = {
    'X_vals': X_vals,
    'V_vals': V_vals,
    'Y_vals': Y_vals,
    'Z_vals': Z_vals,
    'action': action_list # actionのリスト
}
print(len(data["X_vals"]))
df = pd.DataFrame(data)

# CSVファイルとして保存
df.to_csv(r'C:\Users\User\Documents\B4輪講\photo_data\hiv_simulation_initial_results.csv', index=False)