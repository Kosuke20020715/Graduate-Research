import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


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

actions = np.arange(0, 1.1, 0.1)  # 行動 (投薬量u)

#NNの定義
class DQN(nn.Module):#pytorchではnn.Moduleを継承する
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__() #親の初期化
        self.fc1 = nn.Linear(input_dim, 16)#全結合層
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) #最終層を返す

#エージェントの定義
class DQNAgent:
    def __init__(self, state_dim, action_dim):#初期化, ハイパーパラメータの設定
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=1000) #最長1000の経験をメモリに保存
        self.gamma = 0.99  # 割引率
        self.epsilon = 1.0  # 探索の確率
        self.epsilon_decay = 0.995  # 探索確率の減少率
        self.epsilon_min = 0.01
        self.learning_rate = 0.01#学習率
        self.batch_size = 32 #バッチサイズ
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)#最適化アルゴリズム　,lr:学習係数
        self.criterion = nn.MSELoss() #ロス関数, 平均二乗誤差

    def remember(self, state, action, reward, next_state, done): #経験の保存
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): #ε-Greedy法による行動選択
        if np.random.rand() <= self.epsilon: #[0,1]の間でランダムに生成した数がε以下➡ランダム
            return random.sample([0,1],1)[0]
        state = torch.FloatTensor(state).unsqueeze(0)#stateをテンソルに変換　次元を追加
        act_values = self.model(state) #状態をNNに入力/Q値を計算
        return torch.argmax(act_values).item() #最大のQ値をもつ行動(index)を返す(活用)

    #過去の経験を再学習しモデルをトレーニング(バッファ)
    def replay(self):
        if len(self.memory) < self.batch_size: #memoryの数がバッチサイズ未満➡学習の停止(経験不足のため)
            return

        minibatch = random.sample(self.memory, self.batch_size) #ランダムにバッファから経験をサンプリングし, 利用する
        #sample(引用先のリスト, 引用個数)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done: #エピソードが完了していない場合
                target = reward + self.gamma * torch.max(self.model(next_state)).item() #Q値の更新
            target_f = self.model(state) #モデルの予測値
            target_f[0][action] = target
            self.optimizer.zero_grad()#勾配の初期化
            loss = self.criterion(target_f, self.model(state))#ロス関数による正解との誤差の計算
            loss.backward()#誤差逆伝搬
            self.optimizer.step()#パラメータの更新

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#トレーニング
def train_dqn(episodes=500):
    agent = DQNAgent(state_dim=4, action_dim=len(actions))
    X, V, Y, Z = 1000, 1, 1, 0.01
    X_vals, V_vals, Y_vals, Z_vals = [], [], [], []
    for e in range(episodes):
        state = [X, V, Y, Z]
        done = False
        total_reward = 0
        max_steps = 200 #ステップ数
        steps = 0 #初期のステップ
        while not done:
            action_idx = agent.act(state)
            action = actions[action_idx]
            next_state = hiv_model(X, V, Y, Z, action)
            X_new, V_new, Y_new, Z_new = next_state
            reward = np.log(V / V_new) - (action - action_idx) * np.log(action + 1e-8)
            total_reward += reward
            done = steps >= max_steps #最大ステップで終了
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            X, V, Y, Z = X_new, V_new, Y_new, Z_new
            steps += 1
            
        X_vals.append(X)
        V_vals.append(V)
        Y_vals.append(Y)
        Z_vals.append(Z)
        
        agent.replay()
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2}")

    return X_vals, V_vals, Y_vals, Z_vals

X_vals, V_vals, Y_vals, Z_vals = train_dqn()


plt.figure(figsize=(6, 10))
plt.subplot(4, 1, 1)
plt.plot(X_vals, label='X (Healthy CD4+ T cells)')
plt.xlabel('days')
plt.ylabel('X')
plt.title('X over 500 Episodes')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(V_vals, label='V (free Virus)', color='orange')
plt.xlabel('days')
plt.ylabel('V')
plt.title('V over 500 Episodes')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(Y_vals, label='Y (infected Virus)', color='red')
plt.xlabel('days')
plt.ylabel('Y')
plt.title('Y over 500 Episodes')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(Z_vals, label='Z (Virus)', color='green')
plt.xlabel('days')
plt.ylabel('Z')
plt.title('Z over 500 Episodes')
plt.legend()

plt.tight_layout()
plt.show()

