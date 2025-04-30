import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import deque
import torch.nn.functional as F
from statistics import mean

device = torch.device("cuda" if torch.cuda. is_available() else "cpu")#GPUへの切り替え
print(device)

def hiv_model(X, V, Y, Z, action, dt=0.05):
    # Parameters
    lambda_, mu_v, mu_y, mu_x, mu_z = 10, 2.4, 0.02, 0.02, 0.24
    r, P_max, k1, k2, N = 0.03, 1500, 2.4e-05, 0.0030, 1200

    # Vectorized updates
    dX = lambda_ / (1 + V) - mu_x * X + r * X * (1 - (X + Y + Z) / P_max) - k1 * X * V
    dY = k1 * X * V - mu_y * Y - k2 * Y
    dZ = k2 * Y - mu_z * Z
    dV = (1 - action) * N * mu_z * Z - k1 * X * V - mu_v * V

    X_new = np.minimum(X + dX * dt, 1500)
    V_new = V + dV * dt
    Y_new = Y + dY * dt
    Z_new = Z + dZ * dt

    return [X_new, V_new, Y_new, Z_new]

#患者のデータ(csv)の読み込み
patient_data = pd.read_csv(r"C:\Users\redsw\OneDrive\デスクトップ\輪講\python_data\Q_learning\Research\csvdata\newcombine.csv")

#投薬の種類をラベリング
# 列 "APV" から右の列を投薬データとして抽出
medication_cols = patient_data.columns[patient_data.columns.get_loc('APV'):patient_data.columns.get_loc('RAL')]

# 投薬データからactionを生成する関数
def generate_actions(row):
    # 投与された薬をチェックしてラベリング
    medications = []
    for i, col in enumerate(medication_cols, 1):
        if row[col] > 0:  # 投薬があれば
            medications.append(i)  # ラベルは 1, 2, ..., 22

    # 投薬がなければ 0 を追加
    while len(medications) < 5:
        medications.append(0)

   # 0が最初に出現した位置を取得し、それ以降をすべて0にする
    if 0 in medications:
        first_zero_index = medications.index(0)
        # 0の後には薬が入っていないはずなので、それ以降を0に置き換える
        for i in range(first_zero_index + 1, len(medications)):
            medications[i] = 0
    # 0以外の薬の数を数える
    non_zero_count = sum(1 for med in medications if med != 0)
    
    # 正規化: 0以外の薬の数を最大5として正規化
    normalized_value = non_zero_count / 5
    
    # 正規化した値をリストの右端に追加
    medications.append(normalized_value)  
    return medications

# 状態（state）および次の状態（next_state）の正規化用の関数
def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# データの最小値と最大値を手動で指定（CD4とVLoadの範囲）
CD4_min, CD4_max = 0, 1500  # CD4 countの範囲例
VLoad_min, VLoad_max = 1e-6, 1e2  # ウイルス量の範囲例

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
offline_memory = ReplayMemory(100000)
online_memory = ReplayMemory(100000)
action_list = [0, 0.2, 0.4, 0.6, 0.8, 1] #actionの選択肢

# 状態数と行動数(Offline)
state_size = 5  # CD4とRNA,subtype,start date,stop date
action_size = 23**5 # 投薬の種類

#状態数と行動数(Online)
online_action_size = 6
epsilon_start = 1.0  # 初期値
epsilon_end = 0.01  # 終了時の値
epsilon_decay = 0.01  # 減衰率
epsilon = epsilon_start

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
from copy import deepcopy

# ネットワークの初期化 (グローバル)
online_policy_net = DQN(state_size, online_action_size).to(device)
online_target_net = deepcopy(online_policy_net).to(device)

offline_policy_net = DQN(state_size, action_size).to(device)
offline_target_net = deepcopy(offline_policy_net).to(device)

# オンライン強化学習用のオプティマイザ
online_optimizer = optim.Adam(online_policy_net.parameters(), lr=0.0001)

# オフライン強化学習用のオプティマイザ
offline_optimizer = optim.Adam(offline_policy_net.parameters(), lr=0.0001)
BATCH_SIZE = 32

def multidim_to_1d(actions): #31進数から10進数
    decimal_values = []
    for action in actions:
        decimal_value = 0
        for i, a in enumerate(reversed(action)):
            decimal_value += a * (23 ** i)  # 各桁を23進数から10進数に変換
        decimal_values.append(decimal_value)
    return torch.tensor(decimal_values, dtype=torch.int64).to(device)

def decimal_to_base30(value): #10進数から31進数
    if value == 0:
        return [0,0,0,0,0]  # 0の場合は0を返す
    base30_digits = []
    while value > 0:
        base30_digits.append(value % 23)  # 25で割った余りを取得
        value //= 23  # 31で割る
     # 長さが5になるように前方に0を追加
    while len(base30_digits) < 5:
        base30_digits.insert(0, 0)  # 先頭に0を挿入
    base30_digits = base30_digits[::-1]  # 桁を逆にして返す
    return base30_digits

def decode_subtype(subtype): #subtypeの種類を数字にデコードする
    subtype_variety = list(set(patient_data.iloc[:,1].tolist()))
    index = subtype_variety.index(subtype) #入力に対応するindexを出力
    return index

def offline_optimize_model(GAMMA,e,policy_net,target_net, optimizer,state,action,reward,next_state):
    state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
    next_state_tensor = torch.tensor([next_state], dtype=torch.float32).to(device)
    action_tensor = multidim_to_1d([action])  # アクションをテンソルに変換
    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

    Q_values = policy_net(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_Q_values = target_net(next_state_tensor).max(1)[0]

    target = reward_tensor + (GAMMA * next_Q_values)
    loss = F.mse_loss(Q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 20 == 0:
        target_net.load_state_dict(policy_net.state_dict())

def online_optimize_model(GAMMA,e,policy_net,target_net,optimizer,state,action,reward,next_state):
    state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
    next_state_tensor = torch.tensor([next_state], dtype=torch.float32).to(device)
    action_tensor = torch.as_tensor([action], dtype=torch.int64).to(device)
    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
    new_action_tensor = action_tensor.to(dtype=torch.int64).unsqueeze(1)

    Q_values = policy_net(state_tensor).gather(1, new_action_tensor).squeeze(1)

    with torch.no_grad():
        next_Q_values = target_net(next_state_tensor).max(1)[0]

    target = reward_tensor + (GAMMA * next_Q_values)
    loss = F.mse_loss(Q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 20 == 0:
        target_net.load_state_dict(policy_net.state_dict())

#PtIDによるtrain
X_all, V_all, Date_all, actions_all, rewards_all, states_all, next_states_all, csv_actions = [],[],[],[],[],[],[],[]
def process_per_groups():
    count = 0
    csv_actions = []
    
    for group, df_group in patient_data.groupby('PtID'):
        # Define initial values
        states = []
        next_states = []
        # Remove rows with NaN in 'RNADate' to avoid idxmin issues
        df_group = df_group.dropna(subset=['RNADate'])

        if df_group.empty:
            print(f"Skipping group {group} due to missing RNADate values.")
            continue
        min_date_row = df_group.loc[df_group['RNADate'].idxmin()]

        X = min_date_row['CD4Count']
        V = min_date_row['VLoad']
        # start_date = min_date_row['StartDate'] #投与開始日
        # stop_date = min_date_row['StopDate'] #投与終了日
        for index, row in df_group.iterrows():
            # Get current state values
            State_X = row['CD4Count']
            State_V = row['VLoad']
            subtype = decode_subtype(row['Subtype'])
            start_date = row['StartDate'] #開始日
            stop_date = row['StopDate'] #終了日
            # Define current state
            states = [State_X, State_V, subtype, start_date, stop_date]
            states_all.append(states)  # Append only after confirming consistency
            # Determine next state if available
            if index < len(df_group) - 1:
                next_row = df_group.iloc[index + 1]
                nextstate_X = next_row['CD4Count']
                nextstate_V = next_row['VLoad']
                next_start_date = next_row['StartDate']
                next_stop_date = next_row['StopDate']
                next_subtype = decode_subtype(next_row['Subtype'])
                next_states = [nextstate_X, nextstate_V, next_subtype, next_start_date, next_stop_date]
            else:
                # Use the average state as next state for the last row
                mean_values = [np.mean([s[i] for s in states_all]) for i in range(2)] + [subtype] + [start_date] + [stop_date]
                next_states = mean_values
            next_states_all.append(next_states)
            # Generate action and reward
            csv_action = generate_actions(row)
            next_action = generate_actions(next_row) if index < len(df_group) - 1 else [0, 0, 0, 0, 0, 0]
            csv_reward = np.log(State_V / (nextstate_V+1e-8)) - (next_action[5] - csv_action[5]) * np.log(next_action[5] + 1e-8)
            # Add to replay memory and actions/rewards lists
            if csv_action[5] > 1:
                csv_action[5] = 1
            actions_all.append(csv_action[:5])
            rewards_all.append(csv_reward)
            count += 1
            GAMMA = 0
            # Save and optimize model every 100 steps
            if count > BATCH_SIZE and count % 20 ==0 :
                offline_optimize_model( GAMMA, count, offline_policy_net,offline_target_net, offline_optimizer,states,csv_action[:5],csv_reward,next_states)
                online_optimize_model(GAMMA, count, online_policy_net,online_target_net, online_optimizer,states,csv_action[5],csv_reward,next_states)
    torch.save(online_policy_net.state_dict(), r'C:\Users\redsw\OneDrive\デスクトップ\輪講\python_data\Q_learning\Research\torch_net\online_policy_net.pth')
    torch.save(offline_policy_net.state_dict(), r'C:\Users\redsw\OneDrive\デスクトップ\輪講\python_data\Q_learning\Research\torch_net\policy_net.pth')

    return states_all, next_states_all, actions_all, rewards_all, csv_actions
# Run the function
states_all, next_states_all, actions_all, rewards_all, csv_actions = process_per_groups()
# Load trained model for inference
offline_policy_net.load_state_dict(torch.load(r'C:\Users\redsw\OneDrive\デスクトップ\輪講\python_data\Q_learning\Research\torch_net\policy_net.pth'))
online_policy_net.load_state_dict(torch.load(r'C:\Users\redsw\OneDrive\デスクトップ\輪講\python_data\Q_learning\Research\torch_net\online_policy_net.pth'))
subtype = random.choice(patient_data.iloc[:,1].tolist()) #Subtypeをリスト化しランダムに抽出
import numpy 
numpy.set_printoptions(threshold=numpy.inf)
penalty = -10.0  # 適切なペナルティ値を設定
################ここからはオンライン強化学習#################
def action_quantities(decoded_action): #actionの量を出力
    non_zero = sum(1 for med in decoded_action if med != 0) #非零の要素のカウント
    action_quantity = non_zero / 5 #正規化
    return action_quantity

def select_action(state): #onilneでのaction量を求める
    global epsilon
    # print("epsilon: "+str(epsilon))
    if np.random.rand() <= epsilon:  # [0,1]の間でランダムに生成した数がε以下➡ランダム
        return random.choice(range(len(action_list))) 
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # stateをテンソルに変換　次元を追加
        act_values = online_policy_net(state).squeeze()  # 状態をNNに入力/Q値を計算
        return torch.argmax(act_values).item()  # 最大のQ値をもつ行動(index)を返す(活用)

def create_full_action(action,action_index): #リスト形式のactionを作りなおす
    non_zero_count = sum(1 for med in action if med != 0) # 0以外の薬の数を数える
    non_zero_list = action[:non_zero_count] 
    if action_index > non_zero_count:
        action_index = non_zero_count
    random_list = random.sample(non_zero_list,action_index) 
    zero_num = 5 - action_index  #0の要素数
    for i in range(zero_num):
        random_list.append(0)
    return random_list

reward_record = []
total_reward_per_day = [] #1日あたりの報酬の合計
def online_train(X_0, V_0, Subtype, episode): #onlineのトレーニング
    #以下, 1か月に1度X,Vを更新するために用意する初期値(初回のみ適用)
    Y, Z = 1, 0.01
    prev_action = 0 # 初期値
    total_reward = 0
    X_vals, V_vals, actions, total_action, offline_actions, reward_per_day =[], [], [], [], [], []
    global decoded_action, epsilon
    X_vals.append(X_0)
    V_vals.append(V_0)
    X,V = X_0,V_0
    actions.append(prev_action)
    for e in range(0, episode):
        state = [X, V, Subtype, e, e+1] #state+subtype+date/date+7
        action_index = 0 if e == 0 else select_action(next_state) 
        action = 0 if e == 0 else action_list[action_index]  #online
        new_patient_state = torch.tensor([state], dtype=torch.float32).to(device)       
        with torch.no_grad(): #offline actionを計算
            q_values = offline_policy_net(torch.tensor(new_patient_state, dtype=torch.float32)).squeeze()  # 全てのアクションのQ値を取得
            valid_indices = torch.tensor(list(multidim_to_1d(actions_all)), device=device)
            indices = torch.arange(q_values.size(0), device=q_values.device)
            # 各インデックスがvalid_indicesに含まれているかどうかを判定
            is_valid = torch.isin(indices, valid_indices)  # 有効ならTrue、無効ならFalse
            c = is_valid.sum().item() 
            # print(c)
            q_values += penalty * (~is_valid) 
            best_action_idx = q_values.argmax().item()
            decoded_action = decimal_to_base30(best_action_idx)  # indexからの逆変換
        offline_actions.append(decoded_action) 
        full_action = create_full_action(decoded_action, action_index)  # listつきのaction

        actions_all.append(full_action)
        total_action.append(full_action)
        next_state, YZ = hiv_model(X, V, Y, Z, action)[:2], hiv_model(X, V, Y, Z, action)[2:]
        next_state.append(Subtype)
        next_state.append(e)
        next_state.append(e+1)
        Y, Z = YZ
        X_new, V_new = next_state[:2]
        X_vals.append(X_new) 
        V_vals.append(V_new)  
        actions.append(action)
        # print(V, V_new,prev_action, action)
        reward = np.log(V / (V_new+1e-8)) - (action - prev_action) * np.log(action + 1e-8) #報酬計算
        if np.isinf(reward):  # -inf の場合
            if reward_record:  # 過去のrewardが存在する場合
                reward = np.mean(reward_record)  # 平均値に置き換え
            else:
                reward = 0  # デフォルト値（初期状態では0とする）
        X, V = X_new, V_new
        reward_per_day.append(reward/200)
        total_reward += reward
        # print(action_list.index(action))
        GAMMA = 0.5
        if e > BATCH_SIZE and e % 10 == 0:
            offline_optimize_model(GAMMA, e, offline_policy_net, offline_target_net, offline_optimizer,state,decoded_action,reward,next_state) 
            online_optimize_model(GAMMA, e, online_policy_net, online_target_net, online_optimizer,state,action,reward,next_state)
        state = next_state
        prev_action = action 
        # 指数減少
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay * e)
        total_reward_per_day.append(reward_per_day) #1日の平均
        reward_record.append(total_reward) #1人あたりの最終値
    return X_vals, V_vals, actions, reward_record, actions_all, total_action, offline_actions, total_reward_per_day
# 状態数を指定
num_states = 200
state_list = []

for _ in range(num_states):
    state1 = random.randint(0, 1000)  # 0~500の整数
    state2 = round(random.uniform(0, 5.0), 1)  # 0.0~5.0の小数第1位までの浮動小数点数
    state_list.append([state1, state2, decode_subtype(subtype)])

i = 0
X_vals_all, V_vals_all = [], []
for state in state_list:
    epsilon = epsilon_start
    X_vals, V_vals, actions, reward_record, actions_all,total_action, offline_actions, total_reward_per_day = online_train(state[0],state[1],state[2],420)#60日に1度観測可能
    X_vals_all.append(X_vals)
    V_vals_all.append(V_vals)
    i+=1
    print(i)
print(sum(reward_record)/200) #rewardの平均計算
#rewardの平均を求める
def sum_lists_x(lists):
    return [sum(ls) for ls in map(list, zip(*lists))]

total_reward_per_day = sum_lists_x(total_reward_per_day)

# グラフの表示
plt.figure(figsize=(14, 10))

# リストをNumPy配列に変換
X_vals_all = np.array(X_vals_all)
V_vals_all = np.array(V_vals_all)

plt.subplot(2, 2, 1)
i = 0
for i in range(10):
    plt.plot(X_vals_all[i],color='blue')
plt.xlabel('Days')
plt.ylabel('X')

plt.subplot(2, 2, 2)
i = 0
for i in range(10):
    plt.plot(V_vals_all[i],color='orange')
plt.xlabel('Days')
plt.ylabel('V')

# rewardのプロット
plt.subplot(2, 2, 3)
plt.plot(reward_record, color='black')
plt.xlabel('patients')
plt.ylabel('total reward')
plt.title('reward over Episodes')

# rewardのプロット
plt.subplot(2, 2, 4)
plt.plot(total_reward_per_day, color='black')
plt.xlabel('days')
plt.ylabel('reward average')
plt.title('reward over Episodes')

plt.tight_layout()
plt.show()

#full actionを種類に変換して出力
medication_labels = [
    "APV", "IDV", "LPV", "NFV", "RTV", "SQV", "ATV", "TPV", "DRV", "AZT",
    "DDI", "DDC", "D4T", "3TC", "ABC", "FTC", "TDF", "NVP", "DLV", "EFV",
    "ETR", "RAL"
]
# 総日数 
total_days = len(total_action)*7

# プロットの準備
fig, ax = plt.subplots(figsize=(30, 20))

# 投薬スケジュールのプロット
for day_idx, daily_treatments in enumerate(offline_actions):
    start_day = day_idx*7
    for medication in daily_treatments:
        if medication != 0:  # 無投薬の0はプロットしない
            ax.barh(medication, 7, left=start_day, color='#00FFFF')

# 投薬スケジュールのプロット
for day_idx, daily_treatments in enumerate(total_action):
    start_day = day_idx *7
    for medication in daily_treatments:
        if medication != 0:  # 無投薬の0はプロットしない
            ax.barh(medication, 7, left=start_day, color='blue')

# グラフの装飾
ax.set_yticks(range(1, 23))
ax.set_yticklabels(medication_labels)
ax.set_xticks(np.arange(0, total_days + 1, 28))
ax.set_xticklabels(np.arange(0, total_days + 1, 28))
ax.set_xlabel("Days")
ax.set_ylabel("Medication Type")
ax.set_title("PtID: Example - 70-Days Medication Schedule")

plt.grid()
plt.show()

#ファイルの削除
import os
os.remove(r'C:\Users\redsw\OneDrive\デスクトップ\輪講\python_data\Q_learning\Research\torch_net\policy_net.pth')
print(sum(total_reward_per_day)/420)