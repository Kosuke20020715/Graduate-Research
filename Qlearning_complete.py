import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
############################################治療無しの場合のコード#########################################################
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

# 状態空間と行動空間の定義
X_min, X_max, X_interval = 600, 3000, 10
V_min, V_max, V_interval = 0, 5, 0.05
NUM_EPISODES = 500  # 最大試行回数

def equasion(action):
    X_vals_0 = []
    V_vals_0 = []
    Y_vals_0 = []
    Z_vals_0 = []
    steps = 0
    max_steps = 500
    for episode in range(NUM_EPISODES):
        done = False
        while not done:
            if episode == 0 and steps == 0:
                X, V, Y, Z = 1000, 1, 1, 0.01
                X_vals_0.append(X)
                V_vals_0.append(V)
                Y_vals_0.append(Y)
                Z_vals_0.append(Z)

            else:
                observation_next = hiv_model(X, V, Y, Z, action)
                X, V, Y, Z = observation_next  # X,V,Y,Zの更新
                X_vals_0.append(X)
                V_vals_0.append(V)
                Y_vals_0.append(Y)
                Z_vals_0.append(Z)
            
            done = steps >= max_steps
            steps += 1

    return X_vals_0,V_vals_0,Y_vals_0,Z_vals_0

# それぞれの行動に対する結果を取得
X_vals_0, V_vals_0, Y_vals_0, Z_vals_0 = equasion(0)   # action = 0 の場合
X_vals_01, V_vals_01, Y_vals_01, Z_vals_01 = equasion(0.1)  # action = 0.1 の場合


#################################################ここからQ学習###########################################################
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
def decide_action(state, tau):
    a_max = max(q_table[state][:])  # 最大値を計算
    sum_exp_values = sum([np.exp((v - a_max) / tau) for v in q_table[state][:]])
    p = [np.exp((v - a_max) / tau) / sum_exp_values for v in q_table[state][:]]
    if np.any(np.isnan(p)):
        print(f"Warning: NaN detected in probabilities for state {state}")
    action_index = np.random.choice(np.arange(num_actions), p=p)  # インデックスとして行動を選択
    action = actions[action_index]  # 実際の行動を取得
    return action, action_index  # 行動とそのインデックスを返す

# Q学習アルゴリズム
def q_learning():
    X_vals = []
    V_vals = []
    Y_vals = []
    Z_vals = []
    action_list = []
    X_mean = []
    Y_mean = []
    Z_mean = []
    V_mean = []

    X, V, Y, Z = 1000, 1, 1, 0.01
    for episode in range(NUM_EPISODES):
        X_vals.append(X)
        V_vals.append(V)
        Y_vals.append(Y)
        Z_vals.append(Z)

        done = False
        max_steps = 500 #最大のステップ
        steps = 0 #初期のステップ
        while not done:
            state = digitize_state(X, V)  # 初期の状態のインデックス
            tau = 1.0   #初期の計算温度
            action, action_index = decide_action(state,tau)  # 行動とそのインデックスを決める
            # 温度係数の更新(指数ver)
            T_0 = 1.0
            k=0.01
            tau = T_0 * np.exp(-k * episode)
            # tau = tau * 0.999 計算温度の減少(定数ver)
            observation_next = hiv_model(X, V, Y, Z, action) #状態の取得
            X_new, V_new, Y_new, Z_new = observation_next # X,V,Y,Zの更新
            state_next = digitize_state(X_new, V_new)  # 新たな状態のインデックス

            if episode==0 and steps == 0:
                prev_action = 0  #初期の投薬量
                gamma = 0.5 #初期の割引率
            else:
                prev_action = action #行動の更新　
                gamma = 0 #以降の割引率

            reward = np.log(V / V_new) - (action - prev_action) * np.log(action+1e-8)  # actionが0になるのを防ぐ

            q_table[state, action_index] = update_Qtable(state, action_index, reward, state_next, gamma)
            done = steps >= max_steps #最大ステップで終了
            state = state_next
            X, V, Y, Z = X_new, V_new, Y_new, Z_new #状態の更新
            steps += 1 #stepの更新
            action_list.append(action)

        X_vals.append(X)
        V_vals.append(V)
        Y_vals.append(Y)
        Z_vals.append(Z)

        X_mean.append(np.mean(X_vals))
        V_mean.append(np.mean(V_vals))
        Y_mean.append(np.mean(Y_vals))
        Z_mean.append(np.mean(Z_vals))



    return X_mean, V_mean, Y_mean, Z_mean ,action_list

X_mean, V_mean, Y_mean, Z_mean, action_list = q_learning()

################################################################ここまでQ学習#########################################################

# グラフ全体のサイズを指定（幅10インチ、高さ12インチに設定）
plt.figure(figsize=(10, 10))

# Xのプロット
plt.subplot(2, 2, 1)
plt.plot(X_vals_0, label='X (Healthy CD4+ T cells) action=0',color='blue')
plt.plot(X_vals_01, label='X (Healthy CD4+ T cells) action=0.1', linestyle='--',color='blue')
plt.plot(X_mean, label='X (Healthy CD4+ T cells) Q-learning', linestyle='-.',color='blue')
plt.xlabel('days')
plt.ylabel('X')
plt.title('X over 500 Episodes')
plt.legend()

# Vのプロット
plt.subplot(2, 2, 2)
plt.plot(V_vals_0, label='V (free Virus) action=0', color='orange')
plt.plot(V_vals_01, label='V (free Virus) action=0.1', color='orange', linestyle='--')
plt.plot(V_mean, label='V (free Virus) Q-learning', color='orange', linestyle='-.')
plt.xlabel('days')
plt.ylabel('V')
plt.title('V over 500 Episodes')
plt.legend()

# Yのプロット
plt.subplot(2, 2, 3)
plt.plot(Y_vals_0, label='Y (infected Virus) action=0', color='red')
plt.plot(Y_vals_01, label='Y (infected Virus) action=0.1', color='red', linestyle='--')
plt.plot(Y_mean, label='Y (infected Virus) Q-learning', color='red', linestyle='-.')
plt.xlabel('days')
plt.ylabel('Y')
plt.title('Y over 500 Episodes')
plt.legend()

# Zのプロット
plt.subplot(2, 2, 4)
plt.plot(Z_vals_0, label='Z (Virus) action=0', color='green')
plt.plot(Z_vals_01, label='Z (Virus) action=0.1', color='green', linestyle='--')
plt.plot(Z_mean, label='Z (Virus) Q-learning', color='green', linestyle='-.')
plt.xlabel('days')
plt.ylabel('Z')
plt.title('Z over 500 Episodes')
plt.legend()

plt.tight_layout()
plt.show()

# データをcsvに保存

data = {
    'X_vals': X_mean,
    'V_vals': V_mean,
    'Y_vals': Y_mean,
    'Z_vals': Z_mean,
    'action': action_list # actionのリスト
}
print(len(data["X_vals"]))
df = pd.DataFrame(data)

# CSVファイルとして保存
# df.to_csv(r'C:\Users\User\Documents\B4輪講\photo_data\hiv_simulation_complete_results.csv', index=False)