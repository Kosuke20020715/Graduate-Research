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
    X_new = min(X + dX * dt,1500)
    V_new = V + dV * dt
    Y_new = Y + dY * dt
    Z_new = Z + dZ * dt

    observation = [X_new, V_new, Y_new, Z_new]
    return observation

# 状態空間と行動空間の定義
X_min, X_max, X_interval = 0, 1500, 10
V_min, V_max, V_interval = 0, 5, 0.05
actions = np.arange(0, 1.1, 0.1)
num_actions = len(actions)
q_table = np.random.uniform(low=-1, high=1, size=(150 * 100, num_actions))

# binsの定義
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 状態を離散化して一意の整数に変換する関数
def digitize_state(X, V):
    digitized = [
        np.digitize(X, bins=bins(X_min, X_max, 150)),
        np.digitize(V, bins=bins(V_min, V_max, 100))
    ]
    digital_num = (digitized[0]+1)*(digitized[1]+1)
    return digital_num

# Q値の更新式
def update_Qtable(state, action, reward, state_next, gamma):
    alpha = 0.5
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

reward_list, total_reward_per_day = [], [] #報酬の蓄積
# Q学習アルゴリズム
def q_learning(X, V, Y, Z):
    X_vals, V_vals, Y_vals, Z_vals, action_list, reward_per_day = [], [], [], [], [], []
    total_rewards = 0 
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
            k=0.01
            tau = T_0 * np.exp(-k * episode)

            observation_next = hiv_model(X, V, Y, Z, action)
            X_new, V_new = observation_next[0:2]

            state_next = digitize_state(X_new, V_new)
            prev_action = 0        
            
             # 報酬の計算(追記)
            # print(np.log(action+1e-8))
            reward = 0
            reward_per_day.append(reward/200)
            total_rewards += reward #報酬の加算

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
            k=0.01
            tau = T_0 * np.exp(-k * episode)

            observation_next = hiv_model(X, V, Y, Z, action)
            X_new, V_new, Y, Z = observation_next

            state_next = digitize_state(X_new, V_new)

            # print(np.log(action+1e-8))
            #報酬の計算
            # print(V, V_new, action, prev_action)
            reward =  np.log(V / (V_new)) - (action - prev_action) * np.log(action+1e-8)
            
            if np.isinf(reward):  # -inf の場合
              if reward_list:  # 過去のrewardが存在する場合
                  reward = np.mean(reward_list)  # 平均値に置き換え
              else:
                  reward = 0  # デフォルト値（初期状態では0とする）
            reward_per_day.append(reward/200)
            total_rewards += reward #報酬の加算

            prev_action = action

            q_table[state, action_index] = update_Qtable(state, action_index, reward, state_next, 0)

            state = state_next
            X, V = X_new, V_new

            X_vals.append(X)
            V_vals.append(V)
            Y_vals.append(Y)
            Z_vals.append(Z)

    reward_list.append(total_rewards)
    total_reward_per_day.append(reward_per_day)

    return X_vals, V_vals, Y_vals, Z_vals, action_list, reward_list, total_reward_per_day

NUM_EPISODES = 420
state_list = [] 

for _ in range(200):
    state1 = random.randint(0, 1000)
    state2 = round(random.uniform(0.0, 5.0), 1)
    state_list.append([state1, state2])

X_vals_all, V_vals_all = [], []

# 200個の初期条件でシミュレーションを実行
for state in state_list:
    Y, Z = 1.0, 0.01
    X_vals, V_vals, Y_vals, Z_vals, action_list, reward_list, total_reward_per_day = q_learning(state[0], state[1], Y, Z)
    X_vals_all.append(X_vals)
    V_vals_all.append(V_vals)


X_vals_all = np.array(X_vals_all)
V_vals_all = np.array(V_vals_all)

#rewardの平均を求める
def sum_lists_x(lists):
    return [sum(ls) for ls in map(list, zip(*lists))]

total_reward_per_day = sum_lists_x(total_reward_per_day)

print(sum(reward_list)/200)
print(sum(total_reward_per_day)/140)
# グラフの表示
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(reward_list, color='black')
plt.xlabel('patients')
plt.ylabel('reward')
plt.title('reward over 200 patients')

# rewardのプロット
plt.subplot(2, 2, 2)
plt.plot(total_reward_per_day, color='black')
plt.xlabel('days')
plt.ylabel('reward average')
plt.title('reward over Episodes')

# Xのプロット
plt.subplot(2, 2, 3)
i=0
for i in range(10):
    plt.plot(X_vals_all[i],color='blue')
plt.xlabel('days')
plt.ylabel('X')
plt.title('X over 500 Episodes')

# Vのプロット
plt.subplot(2, 2, 4)
i=0
for i in range(10):
    plt.plot(V_vals_all[i],color='orange')
plt.xlabel('days')
plt.ylabel('V')
plt.title('V over 500 Episodes')

plt.tight_layout()
plt.show()
# # Yのプロット
# plt.subplot(2, 2, 3)
# for Y_vals in sample_Y_vals:
#     plt.plot(Y_vals, color='red', )
# plt.xlabel('days')
# plt.ylabel('Y')
# plt.title('Y over 500 Episodes')

# # Zのプロット
# plt.subplot(2, 2, 4)
# for Z_vals in sample_Z_vals:
#     plt.plot(Z_vals, color='green', )
# plt.xlabel('days')
# plt.ylabel('Z')
# plt.title('Z over 500 Episodes')

# plt.tight_layout()
# plt.show()

# #actionのプロット
# plt.plot(action_list)
# plt.xlabel('days')
# plt.ylabel('u')
# plt.title('u over 500 Episodes')
# plt.show()

