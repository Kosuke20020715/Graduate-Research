import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import deque
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda. is_available() else "cpu")#GPUへの切り替え
# print(device)
#患者のデータ(csv)の読み込み
patient_data = pd.read_csv(r"C:\Users\redsw\OneDrive\デスクトップ\輪講\python_data\Q_learning\Research\csvdata\newcombine.csv")

#投薬の種類をラベリング
# 列 "APV" から右の列を投薬データとして抽出
medication_cols = patient_data.columns[patient_data.columns.get_loc('APV'):]

# 投薬データからactionを生成する関数
def generate_actions(row):
    # 投与された薬をチェックしてラベリング
    medications = []
    for i, col in enumerate(medication_cols, 1):
        if row[col] > 0:  # 投薬があれば
            medications.append(i)  # ラベルは 1, 2, ..., 30

    # 投薬がなければ 0 を追加
    while len(medications) < 6:
        medications.append(0)

   # 0が最初に出現した位置を取得し、それ以降をすべて0にする
    if 0 in medications:
        first_zero_index = medications.index(0)
        # 0の後には薬が入っていないはずなので、それ以降を0に置き換える
        for i in range(first_zero_index + 1, len(medications)):
            medications[i] = 0
    # 0以外の薬の数を数える
    non_zero_count = sum(1 for med in medications if med != 0)
    
    # 正規化: 0以外の薬の数を最大6として正規化
    normalized_value = non_zero_count / 6
    
    # 正規化した値をリストの右端に追加
    medications.append(normalized_value)  
    
    return medications

# 状態（state）および次の状態（next_state）の正規化用の関数
def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# データの最小値と最大値を手動で指定（CD4とVLoadの範囲）
CD4_min, CD4_max = 0, 1500  # CD4 countの範囲例
VLoad_min, VLoad_max = 1e-8, 1e6  # ウイルス量の範囲例

# 状態を正規化する関数
def normalize_state(state):
    # state = [CD4, VLoad]
    CD4 = state[0]
    VLoad = state[1]
    
    # CD4CountとVLoadをそれぞれ正規化
    normalized_CD4 = min_max_scale(CD4, CD4_min, CD4_max)
    normalized_VLoad = min_max_scale(VLoad, VLoad_min, VLoad_max)
    
    return [normalized_CD4, normalized_VLoad]

# 行動インデックスの逆変換関数
def decode_action(action_idx, medication_options=31, num_meds=6):
    action = []
    for i in range(num_meds):
        action.append(action_idx % medication_options)
        action_idx //= medication_options
    return action

# リプレイメモリの作成
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# メモリにデータを保存する
memory = ReplayMemory(100)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
# 状態数と行動数
state_size = 2  # CD4とRNA
action_size = 31*31*31*31*31*31 # 投薬の種類

# DQNエージェントの初期化
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

BATCH_SIZE = 8
GAMMA = 0.5

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))
    
    # 正規化された状態と次の状態を作成
    states_batch = torch.tensor(np.array([normalize_state(s) for s in batch[0]]), dtype=torch.float32)
    next_states_batch = torch.tensor(np.array([normalize_state(ns) for ns in batch[3]]), dtype=torch.float32)
    
    actions_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
    rewards_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)

    # Q(s_t, a) 計算
    state_action_values = policy_net(states_batch).gather(1, actions_batch)

    # Q(s_{t+1}, a) for the next state
    next_state_values = target_net(next_states_batch).max(1)[0].detach().unsqueeze(1)

    # 期待されるQ値
    expected_state_action_values = rewards_batch + (GAMMA * next_state_values)

    # 損失計算
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# オフライン強化学習のトレーニングループ
for i_episode in range(1000):
    optimize_model()

#PtIDによるtrain
X_all, Y_all, Z_all, V_all, Date_all, actions_all, rewards_all, states_all, next_states_all, csv_actions = [],[],[],[],[],[],[],[],[],[]
def process_per_groups():
    for group, df_group in patient_data.groupby('PtID'):  # IDごとの処理
        # 各グループの状態(state)・行動(action)・報酬(reward)を定義
        states = []
        next_states = []
        actions = []
        actions.append([0,0,0,0,0,0,0])
    # RNADate列で最小の日付を持つ行の抽出
        min_date_row = df_group.loc[df_group['RNADate'].idxmin()]
        min_date = df_group.iloc[0, 1] #開始時間

        # X, V, Y, Zの初期値の設定
        X = min_date_row['CD4Count']
        V = min_date_row['VLoad'] # 初期値(ここはデータセットの初期値を用いて変化させるべき)

        X_vals, V_vals, Y_vals, Z_vals, Date = [], [], [], [], [] #状態の初期条件をリストに追加
        X_vals.append(X)
        V_vals.append(V)
        Date.append(min_date)
        # print(f"Group: {group} -> Start RNADate: {start_num}, Final RNADate: {final_num}")
        
        # StartDateからStopDateの間、1日ごとにシミュレーション
        # groupごとにindexをリセット
        df_group['group_index'] = df_group.groupby('PtID').cumcount()
        # 'group_index'を本当のindexに上書き
        df_group = df_group.set_index('group_index')
        
        # 投薬データと投薬量に基づいてラベルを生成/actionの定義
        for index, row in df_group.iterrows():
            State_X = row['CD4Count'] #csvのX
            State_V = row['VLoad'] #csvのV
            index = df_group.index.get_loc(index) 

            # 次の行のstateを取得 (index+1の行)
            if index < len(df_group) - 1:
                next_row = df_group.iloc[index + 1]
                nextstate_X = next_row['CD4Count']  # 次のCD4値
                nextstate_V = next_row['VLoad']  # 次のウイルス量
                next_action = generate_actions(next_row)
            else:
                # 最後の行の場合、next_stateをゼロベクトルやNoneなどで処理
                nextstate_X = 0  # または None など適切な値
                nextstate_V = 1e-8  # または None など適切な値
                next_action = [0,0,0,0,0,0,0]

            states=[State_X,State_V] #statesのリストへ追加
            next_states=[nextstate_X,nextstate_V] #nextstateのリストへ追加
            csv_action = generate_actions(row) #実際のデータセットのaction
            # print(modify_dim(csv_action))
            csv_reward = np.log(State_V / (nextstate_V)) - (next_action[6] - csv_action[6]) * np.log(csv_action[6] + 1e-8)

        memory.push(states, csv_action[:6], csv_reward, next_states)
        # print(states, csv_action[:6], csv_reward, next_states)

    return X_all, V_all, Y_all, Z_all, Date_all, actions_all, states_all, next_states_all, rewards_all, csv_actions

X_all, V_all, Y_all, Z_all, Date_all, actions_all, states_all, next_states_all, rewards_all, csv_actions = process_per_groups()
#X_allなどの整形
# X_all, V_all, Y_all, Z_all = np.array(X_all).reshape(-1,1),np.array(V_all).reshape(-1,1),np.array(Y_all).reshape(-1,1),np.array(Z_all).reshape(-1,1)
for i, (dates, series) in enumerate(zip(Date_all, X_all)):
    plt.plot(dates, series)

# 患者の状態例(初期状態が既知の場合)
new_patient_state = torch.tensor([[250, 1.0]], dtype=torch.float32)

# 行動の選択（epsilon-greedyによる行動選択も可能）
with torch.no_grad():
    q_values = policy_net(new_patient_state).squeeze() # 全てのアクションのQ値を取得
    best_action_idx = q_values.argmax().item()
    decoded_action = decode_action(best_action_idx)
print(f"推奨される治療方針 (アクション): {best_action_idx}{decoded_action}")

