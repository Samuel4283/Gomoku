# torch, nn, optim：PyTorch 中建立神經網路與訓練模型的核心模組
import torch
import torch.nn as nn
import torch.optim as optim

# 用於隨機選擇、數值處理
import numpy as np
import random

# Python 提供的雙向佇列，這裡用來實作 經驗回放記憶庫（Replay Buffer）
from collections import deque

"""
    DQN 網路架構設計
    建立一個 Q-Network
    輸入：棋盤狀態
    輸出：每個格子對應的 Q 值
"""
class DQN(nn.Module): # NN 初始模型
    def __init__(self, board_size, hidden_dim=128): # 影藏層 128
        super(DQN, self).__init__()
        self.fc = nn.Sequential( # Fully Connected
            nn.Flatten(),                                       # 方便接進全連接層
            nn.Linear(board_size * board_size, hidden_dim),     # 輸入層 把棋盤轉成特徵向量
            nn.ReLU(),                                          # 激活函數 增加非線性表達能力
            nn.Linear(hidden_dim, hidden_dim),                  # 隱藏層 讓模型能學到更深的策略
            nn.ReLU(),
            nn.Linear(hidden_dim, board_size * board_size)      # 輸出層：對每個格子產生一個 Q 值，用來判斷哪個動作比較好。
        )

    # forward 定義了資料前向傳遞的路徑，輸入 state → 輸出 Q 值向量。
    def forward(self, x): # 在 x 的位置下棋 
        return self.fc(x)

# 這是「玩家」，負責學習與下決策
class DQNAgent:
    """
        DQN Agent：定義行為、學習方式
        lr: 學習率
    """
    def __init__(self, board_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995): # 訓練參數
        self.board_size = board_size

        # 建立 主 Q 網路 與 目標 Q 網路  DQN 的穩定訓練技巧：更新是根據 target model 的 Q 值，避免不穩定。
        self.model = DQN(board_size)
        self.target_model = DQN(board_size)
        self.target_model.load_state_dict(self.model.state_dict())

        #  使用 Adam 最佳化器來訓練模型參數。
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 記錄探索策略與衰減參數，實作 ε-greedy 策略用。
        self.gamma = gamma # 折扣因子，決定未來回報的重要性
        self.epsilon = epsilon # 探索率，決定是否隨機探索  
        self.epsilon_min = epsilon_min # 探索的最低限度
        self.epsilon_decay = epsilon_decay # 每次學習後 epsilon 衰減多少


        self.memory = deque(maxlen=10000) #  經驗回放記憶庫，最多保留一萬筆資料。
        self.batch_size = 64 # 每次 replay 從記憶庫抽多少資料來訓練（一般介於 32~128）


    """
        將一回合的經驗 (s, a, r, s', done) 加入記憶庫。
    """
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """
        決策策略：ε-greedy
    """
    def act(self, state, valid_actions):
        # 以 epsilon 的機率隨機選動作（探索）
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # 否則用主網路預測每個動作的 Q 值（利用）
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        q_values = q_values.detach().numpy().flatten()

        # 選擇有效合法動作中 Q 值最高者執行
        valid_q_values = [(a, q_values[a]) for a in valid_actions]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]

        # 回傳這個棋盤中 預測最高 Q 值
        return best_action

    """
        經驗回放：訓練模型
    """
    def replay(self):
        # 果記憶庫資料不足，先不要學
        if len(self.memory) < self.batch_size:
            return

        # 隨機抽出一批經驗資料（抽樣訓練）
        minibatch = random.sample(self.memory, self.batch_size)

        # 對每一組經驗算目標 Q 值：target = r + γ * max(Q(s',a'))
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.model(state_tensor)[0].detach().clone()

            # 如果遊戲結束，就用 reward；否則加上未來最好的 Q 值
            if done:
                target[action] = reward
            else:
                next_q = self.target_model(next_state_tensor)[0]
                target[action] = reward + self.gamma * torch.max(next_q).item()

            # 損失函數是 MSE：模型預測與目標值的差距
            output = self.model(state_tensor)[0]
            loss = nn.MSELoss()(output, target) # 目標：讓 Q 網路預測值趨近於 reward + γ * max(Q(s', a'))
            
            # 執行反向傳播 + 更新參數
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 探索率衰減，逐漸從探索 → 利用
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    """
        更新目標模型（定期同步）
    """
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
