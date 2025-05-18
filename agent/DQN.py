import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

"""
    DQN 網路架構設計
"""
class DQN(nn.Module): # NN 初始模型
    def __init__(self, board_size, hidden_dim=128): # 放入棋盤大小 影藏層 128
        super(DQN, self).__init__()
        self.fc = nn.Sequential( # Fully Connected
            nn.Flatten(),                                       # 方便接進全連接層
            nn.Linear(board_size * board_size, hidden_dim),     # 輸入層 把棋盤轉成特徵向量
            nn.ReLU(),                                          # 激活函數 增加非線性表達能力
            nn.Linear(hidden_dim, hidden_dim),                  # 隱藏層 讓模型能學到更深的策略
            nn.ReLU(),
            nn.Linear(hidden_dim, board_size * board_size)
        )

    def forward(self, x): # 在 x 的位置下棋
        return self.fc(x)


class DQNAgent:
    """
        | 參數              | 意義                          | 為什麼選這個值
        | --------------- | -------------------------------| -------------------- |
        | `gamma`         | 折扣因子，決定未來獎勵的重要性    | 0.99 → 表示「長遠報酬很重要」
        | `epsilon`       | 探索機率（隨機選 vs. 用模型）     | 初始設 1.0 → 一開始盡量探索
        | `epsilon_min`   | 探索的最低限度                   | 不讓 agent 完全只靠模型，避免卡住
        | `epsilon_decay` | 每次學習後 epsilon 衰減多少      | 慢慢從探索轉向利用
        | `batch_size`    | 每次 replay 拿多少經驗學習       | 越大越穩定，64 是常見經驗值
        | `train_start`   | 記憶庫至少累積多少經驗才開始學習  | 一開始不要立刻學，等有一定量資料

    """
    def __init__(self, board_size, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995): # 訓練參數
        self.board_size = board_size
        self.model = DQN(board_size)
        self.target_model = DQN(board_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min # 探索的最低限度
        self.epsilon_decay = epsilon_decay # 每次學習後 epsilon 衰減多少
        self.memory = deque(maxlen=10000) # Replay Buffer（經驗回放）
        self.batch_size = 64 # 每次 replay 拿多少經驗學習

        """ Replay Buffer（經驗回放） """
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        q_values = q_values.detach().numpy().flatten()
        valid_q_values = [(a, q_values[a]) for a in valid_actions]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]
        return best_action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.model(state_tensor)[0].detach().clone()

            if done:
                target[action] = reward
            else:
                next_q = self.target_model(next_state_tensor)[0]
                target[action] = reward + self.gamma * torch.max(next_q).item()

            output = self.model(state_tensor)[0]
            loss = nn.MSELoss()(output, target) # 目標：讓 Q 網路預測值趨近於 reward + γ * max(Q(s', a'))
            # 用 Adam + 均方差（MSE）進行參數更新 ； DQN 是 Q-learning 的深度版本，核心精神是學習 Q 值
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
