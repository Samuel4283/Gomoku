import numpy as np
import gym
from gym import spaces

""" 建立環境 """
class GomokuEnv(gym.Env):
    def __init__(self, board_size=15, win_length=5):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.win_length = win_length
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(board_size, board_size), dtype=np.int8)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1 for agent, 2 for opponent
        return self.board.copy()

    def step(self, action):
        row, col = divmod(action, self.board_size)

        # 無效落子（例如格子已被佔用）
        if self.board[row, col] != 0:
            return self.board.copy(), -10, True, {"invalid_move": True}

        # 落子
        self.board[row, col] = self.current_player

        # 判斷勝負
        done, win = self.check_win(row, col)

        # 給分
        if done:
            reward = 1 if win else 0
        else:
            reward = 0

        # 換對手（簡單 rule-based 隨機對手）
        if not done:
            self.current_player = 2 if self.current_player == 1 else 1

        return self.board.copy(), reward, done, {}

    def render(self, mode='human'):
        print(self.board)

    def check_win(self, row, col):
        """檢查是否五子連線"""
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                r, c = row, col
                while True:
                    r += dr * d
                    c += dc * d
                    if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_length:
                return True, True  # 遊戲結束，有人獲勝

        if np.all(self.board != 0):
            return True, False  # 平手

        return False, False  # 尚未結束
