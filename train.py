from agent.DQN import DQN, DQNAgent
from GomokuEnv.gomoku_env import GomokuEnv

env = GomokuEnv(board_size=15, win_length=5)
agent = DQNAgent(board_size=15)

episodes = 1000
target_update_freq = 10  # 每 10 回合更新 target network

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
        agent.replay()

        state = next_state
        total_reward += reward

    # 更新 target network
    if episode % target_update_freq == 0:
        agent.update_target_network()

    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
