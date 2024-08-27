import numpy as np
import matplotlib.pyplot as plt

#hiv数理モデル
def hiv_model(X, V, Y, Z, action):
    # パラメータの設定
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

    # 微分方程式に基づく次の状態の計算
    dX = lambda_ / (1 + V) - mu_x * X + r * X * (1 - (X + Y + Z) / P_max) - k1 * X * V
    dY = k1 * X * V - mu_y * Y - k2 * Y
    dZ = k2 * Y - mu_z * Z
    dV = (1 - action) * N * mu_z * Z - k1 * X * V - mu_v * V

    # 時間ステップ分の更新
    dt = 0.01  # 1日ごとのシミュレーション
    X_new = X + dX * dt
    V_new = V + dV * dt
    Y_new = Y + dY * dt
    Z_new = Z + dZ * dt
    

    observation = [X_new, V_new, Y_new, Z_new]

    return observation

NUM_EPISODES = 500  # 最大試行回数

# 状態空間と行動空間の定義
X_min, X_max, X_interval = 600, 3000, 10
V_min, V_max, V_interval = 0, 5, 0.05
actions = np.arange(0, 1.1, 0.1)  # 行動 (投薬量u)
num_actions = len(actions)   # 11
q_table = np.random.uniform(low=0, high=1, size=(240 * 100, num_actions))  # qテーブルの作成,初期化

#binsの定義
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]  # 等差数列の最後と最初を除外

# 状態を離散化して一意の整数に変換する関数
def digitize_state(X, V):
    digitized = [
        np.digitize(X, bins=bins(X_min, X_max, 240)),  # 0~239
        np.digitize(V, bins=bins(V_min, V_max, 100))   # 0~99 
    ]
    digital_num = (digitized[0]+1)*(digitized[1]+1)  # 0~23999
    return digital_num

#Q値の更新式
def update_Qtable(state, action, reward, state_next, gamma):
    alpha = 0.1  # 学習率
    Max_Q_next = max(q_table[state_next][:])  # state_nextの全行の値の中で最大のQ値を返す
    q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * Max_Q_next - q_table[state, action])  # Q値の更新
    return q_table[state, action]

#行動の定義
def decide_action(state,tau):
    sum_exp_values = sum([np.exp(v/tau) for v in q_table[state][:]])  # softmax選択の分母の計算
    p = [np.exp(v/tau)/sum_exp_values for v in q_table[state][:]]   
    if np.any(np.isnan(p)):
        print(f"Warning: NaN detected in probabilities for state {state}")
    action_index = np.random.choice(np.arange(num_actions), p=p)  # インデックスとして行動を選択
    action = actions[action_index]  # 実際の行動を取得
    return action, action_index  # 行動とそのインデックスを返す\

# Q学習アルゴリズム
def q_learning():
    X_vals = []
    V_vals = []
    Y_vals = []
    Z_vals = []
    for episode in range(NUM_EPISODES):
        if episode == 0:
            X, V, Y, Z = 1000, 1, 1, 0.01
            X_vals.append(X)
            V_vals.append(V)
            Y_vals.append(Y)
            Z_vals.append(Z)

            state = digitize_state(X, V)  # 初期の状態のインデックス
            tau = 1.0#初期の計算温度
            action, action_index = decide_action(state,tau)  # 行動とそのインデックスを決める
            tau = tau * 0.99# 計算温度の減少

            observation_next = hiv_model(X, V, Y, Z, action)
            X_new, V_new = observation_next[0:2]  # X,Vの更新

            state_next = digitize_state(X_new, V_new)  # 新たな状態のインデックス
            prev_action = 0
            reward = np.log(V / V_new) - (action - prev_action) * np.log(action+1e-8)  # actionが0になるのを防ぐ

            prev_action = action
            q_table[state, action_index] = update_Qtable(state, action_index, reward, state_next, 0.5)

            state = state_next
            X, V = X_new, V_new
        
        else:
            action, action_index = decide_action(state,tau)  # 行動とそのインデックスを決める
            tau = tau * 0.99#計算温度の調整

            observation_next = hiv_model(X, V, Y, Z, action)
            X_new, V_new, Y, Z = observation_next  # X,V,Y,Zの更新

            state_next = digitize_state(X_new, V_new)  # 新たな状態のインデックス

            reward = np.log(V / V_new) - (action - prev_action) * np.log(action+1e-8)
            print(observation_next)

            prev_action = action

            q_table[state, action_index] = update_Qtable(state, action_index, reward, state_next, 0)

            state = state_next
            X, V = X_new, V_new

        X_vals.append(X)
        V_vals.append(V)
        Y_vals.append(Y)
        Z_vals.append(Z)

    return X_vals, V_vals, Y_vals,Z_vals

X_vals, V_vals, Y_vals, Z_vals = q_learning()


# グラフ全体のサイズを指定（幅10インチ、高さ12インチに設定）
plt.figure(figsize=(6, 10))
# Xのプロット
plt.subplot(4, 1, 1)
plt.plot(X_vals, label='X (Healthy CD4+ T cells)')
plt.xlabel('days')
plt.ylabel('X')
plt.title('X over 500 Episodes')
plt.legend()

# Vのプロット
plt.subplot(4, 1, 2)
plt.plot(V_vals, label='V (free Virus)', color='orange')
plt.xlabel('days')
plt.ylabel('V')
plt.title('V over 500 Episodes')
plt.legend()

# Yのプロット
plt.subplot(4, 1, 3)
plt.plot(Y_vals, label='Y (infected Virus)', color='red')
plt.xlabel('days')
plt.ylabel('Y')
plt.title('Y over 500 Episodes')
plt.legend()

# Zのプロット
plt.subplot(4, 1, 4)
plt.plot(Z_vals, label='Z (Virus)', color='green')
plt.xlabel('days')
plt.ylabel('Z')
plt.title('Z over 500 Episodes')
plt.legend()

plt.tight_layout()
plt.show()

