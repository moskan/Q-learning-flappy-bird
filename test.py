import os
import sys
import pygame
from game.environment import FlappyBirdEnv
from game.q_agent import QLearningAgent

def main():
    # 初始化带渲染的环境，启用UI
    env = FlappyBirdEnv(render_mode="human", enable_ui=True)
    agent = QLearningAgent(action_space=[0, 1], state_space=None)

    # 加载训练好的模型
    model_path = "models/q_flappy_final.pkl"
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        agent.load(model_path)
        agent.epsilon = 0.01  # 测试时使用极小的探索率，几乎完全利用
    else:
        print(f"警告：未找到训练好的模型 {model_path}")
        print("将使用随机策略进行演示...")
        agent.epsilon = 1.0  # 完全随机

    print("\n" + "="*50)
    print("Q-FLAPPY BIRD 演示")
    print("="*50)
    print("游戏控制：")
    print("  SPACE - 开始游戏/重新开始")
    print("  ESC   - 退出游戏")
    print("  R     - 重置游戏")
    print("="*50)
    
    episodes = 5  # 演示5局
    current_episode = 0
    
    # 游戏主循环
    running = True
    while running and current_episode < episodes:
        # 重置环境
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        max_steps = 2000
        
        print(f"\n游戏状态：对局 {current_episode + 1}/{episodes}")
        
        # 等待开始
        while env.game_state == "start" and running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # 按空格键开始游戏
                        env.game_state = "playing"
                        print("  游戏开始！")
                    elif event.key == pygame.K_r:
                        # 按R键重置
                        state = env.reset()
            
            # 渲染当前帧
            env._render()
            pygame.time.delay(16)  # 约60FPS
        
        if not running:
            break
        
        # 游戏进行中
        while env.game_state == "playing" and step < max_steps and running:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            if not running:
                break
            
            # 智能体选择动作
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if done:
                print(f"  游戏结束！分数: {info['score']} | 总奖励: {total_reward:.2f}")
                break
        
        # 游戏结束，等待重新开始或退出
        if env.game_state == "game_over":
            game_over_start_time = pygame.time.get_ticks()
            waiting_for_restart = True
            
            while waiting_for_restart and running:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            # 按空格键开始下一局
                            waiting_for_restart = False
                            current_episode += 1
                        elif event.key == pygame.K_r:
                            # 按R键重置当前局
                            waiting_for_restart = False
                
                if not running:
                    break
                
                # 渲染当前帧
                env._render()
                pygame.time.delay(16)
        
        # 如果因为达到最大步数而退出，也计为一局完成
        if env.game_state == "playing" and step >= max_steps:
            print(f"  达到最大步数！分数: {env.score} | 总奖励: {total_reward:.2f}")
            current_episode += 1
            # 等待一小段时间显示结果
            pygame.time.delay(1000)
    
    env.close()
    print("="*50)
    print("演示结束！")
    print("="*50)

if __name__ == "__main__":
    main()
