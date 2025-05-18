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
