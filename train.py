import torch
from agent.DQN import DQN, DQNAgent
from GomokuEnv.gomoku_env import GomokuEnv
import matplotlib.pyplot as plt

def train_agent(episodes=10000, load_model=False, save_path="models/gomoku_15x15.pth"):
    env = GomokuEnv(board_size=15, win_length=5)
    agent = DQNAgent(board_size=15)
    target_update_freq = 10  # 每 10 回合更新 target network
    rewards = []
    step_count = 0
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
            
            # 有效行動集合
            valid_actions = [i for i in range(15 * 15) if state.flatten()[i] == 0]

            action = agent.act(state, valid_actions)
            next_state, reward, done, info = env.step(action)

            # 若動作無效，給予懲罰並略過該步
            if info.get("invalid_move"):
                reward = -10
                done = True

            agent.remember(state, action, reward, next_state, done)
            # 改成每幾步才訓練一次
            step_count += 1
            if step_count % 4 == 0:
                agent.replay()

            state = next_state
            total_reward += reward

        # 更新 target network
        if (episode+1) % target_update_freq == 0:
            agent.update_target_model()

        if (episode+1) % (target_update_freq*10) == 0:
            torch.save(agent.model.state_dict(), save_path)


        rewards.append(total_reward)
        if episode % 100 == 0:
            plt.plot(rewards)
            plt.savefig("reward_curve.png")

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    train_agent(episodes=10000, load_model=False)