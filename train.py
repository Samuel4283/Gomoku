import torch
from agent.DQN import DQN, DQNAgent
from GomokuEnv.gomoku_env import GomokuEnv

def train_agent(episodes=1000, load_model=False, save_path="model.pth"):
    env = GomokuEnv()
    agent = DQNAgent(15,state_dim=env.state_dim, action_dim=env.action_dim)

    # 載入模型（如有需要）
    if load_model:
        agent.q_network.load_state_dict(torch.load(save_path))
        agent.q_network.train()
        print("載入模型成功，接續訓練...")

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()  # 訓練一下
            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

        # 每 N 回合保存一次模型
        if (episode + 1) % 100 == 0:
            torch.save(agent.q_network.state_dict(), save_path)
            print(f"已保存模型至 {save_path}")

if __name__ == "__main__":
    train_agent(episodes=10000, load_model=False)