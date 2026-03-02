import os
from game.environment import FlappyBirdEnv
from game.q_agent import QLearningAgent

def main():
    # 初始化环境，训练时禁用UI以获得最佳性能
    env = FlappyBirdEnv(render_mode=None, enable_ui=False)
    state_space_size = (11, 21, 3)
    agent = QLearningAgent(action_space=[0, 1],
                           state_space=state_space_size,
                           learning_rate=0.1,
                           discount_factor=0.99,
                           epsilon=1.0,
                           epsilon_decay=0.9995,
                           epsilon_min=0.01)

    # 训练参数
    episodes = 10000
    max_steps_per_episode = 1000
    save_interval = 500
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print("开始训练Q-Flappy Bird...")
    print("="*50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step += 1

        # 每100轮输出一次进度
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"Score: {info['score']:3d} | "
                  f"Total Reward: {total_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f"q_flappy_episode_{episode+1}.pkl")
            agent.save(model_path)

    # 训练结束后保存最终模型
    final_model_path = os.path.join(models_dir, "q_flappy_final.pkl")
    agent.save(final_model_path)
    
    print("="*50)
    print(f"训练完成！最终模型已保存至: {final_model_path}")
    print("="*50)
    env.close()

if __name__ == "__main__":
    main()