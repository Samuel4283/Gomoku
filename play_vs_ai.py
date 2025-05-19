import torch
from agent.DQN import DQNAgent
from GomokuEnv.gomoku_env import GomokuEnv
import numpy as np

def print_board(board):
    symbols = {0: ".", 1: "X", -1: "O"}
    for row in board:
        print(" ".join(symbols[val] for val in row))

def play():
    board_size = 5
    agent = DQNAgent(board_size=board_size)
    env = GomokuEnv(board_size=board_size, win_length=4)

    # 載入訓練好的模型
    agent.q_network.load_state_dict(torch.load("models/gomoku_15x15.pth"))
    agent.q_network.eval()

    state = env.reset()
    done = False

    while not done:
        print_board(env.board)
        print("你的回合（輸入 row col，例如 1 2）：")
        row, col = map(int, input().split())
        action = row * board_size + col

        # 玩家下棋
        next_state, reward, done, info = env.step(action)
        if info.get("invalid_move"):
            print("無效落子，請重試！")
            continue
        if done:
            print_board(env.board)
            print("🎉 你贏了！")
            break

        # AI 下棋
        state = next_state
        valid_actions = [i for i in range(board_size**2) if state.flatten()[i] == 0]
        ai_action = agent.act(state, valid_actions, train=False)
        state, reward, done, info = env.step(ai_action)

        print(f"🤖 AI 落子：({ai_action // board_size}, {ai_action % board_size})")
        if done:
            print_board(env.board)
            print("💀 AI 贏了...")
            break

if __name__ == "__main__":
    play()
