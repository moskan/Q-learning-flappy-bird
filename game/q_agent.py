import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 初始化Q-table。狀態是一個三元組 (horizontal_dist, vertical_dist, velocity_state)
        # 我們用字典來存儲，避免創建過大的稀疏陣列。
        self.q_table = {}

    def _get_q_value(self, state, action):
        """安全地獲取Q值，如果狀態未見過則初始化。"""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """使用epsilon-greedy策略選擇動作。"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)  # 探索
        else:
            # 利用：選擇當前狀態下Q值最大的動作
            q_values = [self._get_q_value(state, a) for a in self.action_space]
            max_q = max(q_values)
            # 如果有多個動作具有相同的最大Q值，隨機選擇一個
            actions_with_max_q = [a for a, q in zip(self.action_space, q_values) if q == max_q]
            return np.random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state, done):
        """執行Q-learning更新。"""
        current_q = self._get_q_value(state, action)
        if done:
            target = reward
        else:
            # 下一個狀態的最大Q值
            max_future_q = max([self._get_q_value(next_state, a) for a in self.action_space])
            target = reward + self.gamma * max_future_q

        # Q值更新公式
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q

        # 衰減epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """保存Q-table到文件。"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
        print(f"模型已保存至: {filepath}")

    def load(self, filepath):
        """從文件加載Q-table。"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data.get('epsilon', self.epsilon_min)
            print(f"模型已加載自: {filepath}")
        else:
            print(f"未找到模型文件 {filepath}，使用初始空模型。")