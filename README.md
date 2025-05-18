# Gomoku
ai training about Gomoku

# structure：
gomoku-dqn/
├── agent/                 # 智能體程式碼 (DQN)
│   ├── dqn.py
│   └── replay_buffer.py
├── env/                   # 環境模擬
│   └── gomoku_env.py
├── models/                # 模型儲存與加載
│   └── model.py
├── train.py               # 訓練主程式
├── evaluate.py            # 評估模型表現
├── play_vs_ai.py          # 玩家與 AI 對戰
└── utils.py               # 工具函式 (紀錄訓練、落子合法性判斷等)

🧩 專案分段實作架構（建議順序）
# 🔹 第 1 部分：環境（GomokuEnv）實作
這是整個強化學習的核心，推薦使用自定義 Gym-like 環境。

內容包含：

棋盤初始化（一般為 15x15 或 10x10）

玩家輪替邏輯

step(action)：處理落子與勝負判斷

reset()：重置環境

render()：可視化（可先省略，後期加）

get_valid_actions()：回傳有效落子位置

check_winner()：判斷勝負或平手

🔧 工具：可使用 gym.Env 為基底來繼承，若不想用 gym，也可以自行定義 class，效果相同但不支援一些便利工具。

# 🔹 第 2 部分：DQN Agent 實作
建立一個能從環境中學習策略的代理模型。

內容包含：

Q-Network 架構（例如 2 層 MLP，或用 CNN）

記憶體 (ReplayBuffer)

ε-greedy 策略探索

Q-learning 更新公式

act(state)、train_step() 等方法

🔧 工具：PyTorch / TensorFlow 都可，推薦 PyTorch 方便調試。

# 🔹 第 3 部分：訓練腳本
撰寫主要訓練流程（loop）。

內容包含：

每一回合循環：

env.reset()

執行 agent.act() → env.step(action) → 儲存結果

每 N 步執行訓練 agent.train_step()

儲存模型 / 檢查學習效果

# 🔹 第 4 部分：UI 或 CLI 對戰介面（可後期整合）
使用 PyQt 或 CLI（純文字）來：

玩家 vs AI 對戰

AI vs AI 自對戰

顯示勝率 / 畫面結果