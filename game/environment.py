import pygame
import numpy as np
import sys
import os

class FlappyBirdEnv:
    def __init__(self, render_mode=None, enable_ui=True):
        # 游戏参数
        self.WIDTH, self.HEIGHT = 400, 600
        self.BIRD_SIZE = 20
        self.PIPE_WIDTH = 50
        self.PIPE_GAP = 250
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -8
        self.PIPE_VELOCITY = -3
        self.GROUND_HEIGHT = 50

        # UI相关属性
        self.enable_ui = enable_ui
        self.game_state = "start"  # "start", "playing", "game_over"
        self.score = 0
        self.high_score = 0
        self.last_action = 0
        self.last_reward = 0
        
        # 游戏对象初始位置
        self.bird_y = self.HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.pipe_spawn_timer = 0
        self.pipe_frequency = 90  # 帧数

        # 颜色定义
        self.COLORS = {
            "sky": (135, 206, 235),  # 天空蓝
            "ground": (222, 184, 135),  # 大地色
            "bird": (255, 255, 0),  # 黄色
            "pipe": (0, 128, 0),  # 绿色
            "pipe_top": (0, 100, 0),  # 深绿色
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "gold": (255, 215, 0),
            "silver": (192, 192, 192),
            "bronze": (205, 127, 50)
        }
        
        # 游戏字体
        self.fonts = {}
        
        # 动画参数
        self.bird_animation = 0
        self.bird_animation_speed = 0.2
        self.title_pulse = 0
        self.title_pulse_speed = 0.05
        
        self.render_mode = render_mode
        if render_mode == "human":
            self._init_pygame()

    def _init_pygame(self):
        """初始化Pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Q-Flappy Bird")
        self.clock = pygame.time.Clock()
        
        # 创建字体
        self._init_fonts()
        
        # 加载声音（可选）
        self.sounds = {}
        self._init_sounds()

    def _init_fonts(self):
        """初始化字体 - 使用Font而不是SysFont避免Windows字体枚举bug"""
        try:
            # 使用pygame.font.Font而不是SysFont
            # None表示使用默认字体，这不会触发系统字体枚举
            self.fonts["title"] = pygame.font.Font(None, 48)
            self.fonts["title_big"] = pygame.font.Font(None, 60)  # 用于标题脉动效果
            self.fonts["subtitle"] = pygame.font.Font(None, 24)
            self.fonts["score"] = pygame.font.Font(None, 36)
            self.fonts["normal"] = pygame.font.Font(None, 20)
            self.fonts["small"] = pygame.font.Font(None, 16)
            
            # 如果需要加粗效果，可以渲染两次并偏移
            # 但为了简单，我们只使用默认字体
        except Exception as e:
            print(f"字体初始化失败: {e}")
            # 无论如何都创建默认字体
            self.fonts["title"] = pygame.font.Font(None, 48)
            self.fonts["title_big"] = pygame.font.Font(None, 60)
            self.fonts["subtitle"] = pygame.font.Font(None, 24)
            self.fonts["score"] = pygame.font.Font(None, 36)
            self.fonts["normal"] = pygame.font.Font(None, 20)
            self.fonts["small"] = pygame.font.Font(None, 16)

    def _init_sounds(self):
        """初始化音效（可选）"""
        pass  # 可以在此添加音效加载代码

    def _get_state(self):
        """将当前游戏状态离散化，供智能体使用。"""
        if not self.pipes:
            next_pipe = [self.WIDTH, self.HEIGHT//2]
        else:
            # 找到最近且尚未通过的管道
            for pipe in self.pipes:
                if pipe['x'] + self.PIPE_WIDTH > self.BIRD_SIZE:
                    next_pipe = [pipe['x'], pipe['gap_y']]
                    break
            else:
                next_pipe = [self.WIDTH, self.HEIGHT//2]

        # 计算离散化状态
        horizontal_dist = max(0, next_pipe[0] - (self.WIDTH//4)) // 20
        horizontal_dist = min(horizontal_dist, 10)  # 离散化到0-10

        vertical_dist = (self.bird_y - next_pipe[1]) // 20
        vertical_dist = max(-10, min(vertical_dist, 10))  # 离散化到-10到10

        velocity_state = 0
        if self.bird_vel < -2:
            velocity_state = -1
        elif self.bird_vel > 2:
            velocity_state = 1

        # 返回一个可哈希的元组作为状态键
        return (int(horizontal_dist), int(vertical_dist), velocity_state)

    def reset(self):
        """重置环境并返回初始状态。"""
        self.bird_y = self.HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.pipe_spawn_timer = 0
        self.game_state = "start" if self.enable_ui else "playing"
        return self._get_state()

    def _handle_events(self):
        """处理Pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
                if self.game_state == "start" and event.key == pygame.K_SPACE:
                    self.game_state = "playing"
                    return 1  # 跳行动作
                    
                elif self.game_state == "game_over" and event.key == pygame.K_SPACE:
                    self.reset()
                    return 1
                    
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                    
                elif event.key == pygame.K_r:
                    self.reset()
                    
        return None

    def step(self, action):
        """
        执行一个动作。
        参数: action (0或1)
        返回: next_state, reward, done, info
        """
        # 处理UI事件
        ui_action = self._handle_events() if self.render_mode == "human" else None
        
        # 如果UI返回了动作（比如空格键开始），则使用UI动作
        if ui_action is not None:
            action = ui_action
        
        # 根据游戏状态处理
        if self.game_state == "playing":
            return self._game_step(action)
        elif self.game_state == "start":
            return self._get_state(), 0, False, {"score": self.score, "state": "start"}
        elif self.game_state == "game_over":
            return self._get_state(), 0, True, {"score": self.score, "state": "game_over"}
            
        return self._get_state(), 0, False, {"score": self.score}

    def _game_step(self, action):
        """游戏进行中的一步"""
        # 1. 处理动作
        self.last_action = action
        if action == 1:  # 跳跃
            self.bird_vel = self.JUMP_STRENGTH

        # 2. 更新物理
        self.bird_vel += self.GRAVITY
        self.bird_y += self.bird_vel
        self.bird_animation += self.bird_animation_speed

        # 3. 更新管道
        for pipe in self.pipes[:]:
            pipe['x'] += self.PIPE_VELOCITY
        self.pipes = [p for p in self.pipes if p['x'] > -self.PIPE_WIDTH]

        # 4. 生成新管道
        self.pipe_spawn_timer += 1
        if self.pipe_spawn_timer >= self.pipe_frequency:
            gap_y = np.random.randint(100, self.HEIGHT - 100 - self.PIPE_GAP)
            self.pipes.append({'x': self.WIDTH, 'gap_y': gap_y, 'passed': False})
            self.pipe_spawn_timer = 0

        # 5. 碰撞检测与奖励计算
        reward = 0.1  # 存活奖励
        done = False

        # 撞地面/天花板
        if self.bird_y <= 0 or self.bird_y >= self.HEIGHT - self.GROUND_HEIGHT:
            reward = -1000
            done = True
            if self.enable_ui:
                self.game_state = "game_over"
                self.high_score = max(self.high_score, self.score)

        # 管道碰撞检测
        bird_rect = pygame.Rect(self.WIDTH//4, self.bird_y, self.BIRD_SIZE, self.BIRD_SIZE)
        for pipe in self.pipes:
            top_pipe = pygame.Rect(pipe['x'], 0, self.PIPE_WIDTH, pipe['gap_y'])
            bottom_pipe = pygame.Rect(pipe['x'], pipe['gap_y'] + self.PIPE_GAP,
                                      self.PIPE_WIDTH, self.HEIGHT)
            if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                reward = -1000
                done = True
                if self.enable_ui:
                    self.game_state = "game_over"
                    self.high_score = max(self.high_score, self.score)
                break
            # 通过管道（计分点）
            if pipe['x'] + self.PIPE_WIDTH < self.WIDTH//4 and not pipe.get('passed', False):
                pipe['passed'] = True
                reward += 10
                self.score += 1

        self.last_reward = reward

        # 6. 获取下一个状态
        next_state = self._get_state()
        info = {'score': self.score, 'state': self.game_state}

        # 7. 可选渲染
        if self.render_mode == "human":
            self._render()

        return next_state, reward, done, info

    def _draw_background(self):
        """绘制游戏背景"""
        # 渐变天空
        for i in range(self.HEIGHT):
            color_ratio = i / self.HEIGHT
            sky_color = (
                int(self.COLORS["sky"][0] * (1 - color_ratio) + 100 * color_ratio),
                int(self.COLORS["sky"][1] * (1 - color_ratio) + 149 * color_ratio),
                int(self.COLORS["sky"][2] * (1 - color_ratio) + 237 * color_ratio)
            )
            pygame.draw.line(self.screen, sky_color, (0, i), (self.WIDTH, i))
        
        # 绘制云朵
        self._draw_clouds()
        
        # 绘制地面
        pygame.draw.rect(self.screen, self.COLORS["ground"], 
                        (0, self.HEIGHT - self.GROUND_HEIGHT, self.WIDTH, self.GROUND_HEIGHT))
        
        # 绘制草地纹理
        for i in range(0, self.WIDTH, 10):
            pygame.draw.line(self.screen, (139, 115, 85), 
                           (i, self.HEIGHT - self.GROUND_HEIGHT), 
                           (i, self.HEIGHT), 1)

    def _draw_clouds(self):
        """绘制云朵"""
        cloud_positions = [
            (50, 100, 60, 30),
            (200, 150, 80, 40),
            (300, 80, 70, 35),
            (350, 200, 60, 30)
        ]
        
        for x, y, width, height in cloud_positions:
            # 让云朵缓慢移动
            x_offset = int(pygame.time.get_ticks() / 100) % self.WIDTH
            x = (x + x_offset) % (self.WIDTH + 100) - 50
            
            # 绘制多层圆形形成云朵
            pygame.draw.ellipse(self.screen, (255, 255, 255, 200), 
                              (x, y, width, height))
            pygame.draw.ellipse(self.screen, (255, 255, 255, 200), 
                              (x + width//3, y - height//2, width//2, height))
            pygame.draw.ellipse(self.screen, (255, 255, 255, 200), 
                              (x + width*2//3, y, width//2, height))

    def _draw_bird(self):
        """绘制小鸟"""
        x = self.WIDTH // 4
        y = int(self.bird_y)
        
        # 小鸟身体
        pygame.draw.circle(self.screen, self.COLORS["bird"], (x, y), self.BIRD_SIZE//2)
        
        # 小鸟眼睛
        eye_radius = self.BIRD_SIZE // 6
        pygame.draw.circle(self.screen, self.COLORS["black"], 
                          (x + self.BIRD_SIZE//4, y - self.BIRD_SIZE//6), eye_radius)
        
        # 小鸟嘴巴
        beak_points = [
            (x + self.BIRD_SIZE//2, y),
            (x + self.BIRD_SIZE//2 + 10, y - 5),
            (x + self.BIRD_SIZE//2 + 10, y + 5)
        ]
        pygame.draw.polygon(self.screen, (255, 165, 0), beak_points)
        
        # 小鸟翅膀（有动画效果）
        wing_offset = int(5 * abs(np.sin(self.bird_animation)))
        wing_points = [
            (x, y),
            (x - 10, y - wing_offset),
            (x, y + 5)
        ]
        pygame.draw.polygon(self.screen, (255, 200, 0), wing_points)

    def _draw_pipes(self):
        """绘制管道"""
        for pipe in self.pipes:
            x = int(pipe['x'])
            gap_y = pipe['gap_y']
            
            # 上管道
            top_pipe_height = gap_y
            # 管道渐变颜色
            for i in range(top_pipe_height):
                color_ratio = i / top_pipe_height
                pipe_color = (
                    int(self.COLORS["pipe"][0] * (1 - color_ratio) + 0 * color_ratio),
                    int(self.COLORS["pipe"][1] * (1 - color_ratio) + 50 * color_ratio),
                    int(self.COLORS["pipe"][2] * (1 - color_ratio) + 0 * color_ratio)
                )
                pygame.draw.line(self.screen, pipe_color, 
                               (x, i), (x + self.PIPE_WIDTH, i), 1)
            
            # 管道顶部
            pygame.draw.rect(self.screen, self.COLORS["pipe_top"], 
                           (x - 5, top_pipe_height - 20, self.PIPE_WIDTH + 10, 20))
            
            # 下管道
            bottom_pipe_top = gap_y + self.PIPE_GAP
            bottom_pipe_height = self.HEIGHT - bottom_pipe_top
            for i in range(bottom_pipe_height):
                color_ratio = i / bottom_pipe_height
                pipe_color = (
                    int(self.COLORS["pipe"][0] * (1 - color_ratio) + 0 * color_ratio),
                    int(self.COLORS["pipe"][1] * (1 - color_ratio) + 50 * color_ratio),
                    int(self.COLORS["pipe"][2] * (1 - color_ratio) + 0 * color_ratio)
                )
                pygame.draw.line(self.screen, pipe_color, 
                               (x, bottom_pipe_top + i), 
                               (x + self.PIPE_WIDTH, bottom_pipe_top + i), 1)
            
            # 管道底部
            pygame.draw.rect(self.screen, self.COLORS["pipe_top"], 
                           (x - 5, bottom_pipe_top, self.PIPE_WIDTH + 10, 20))

    def _draw_start_screen(self):
        """绘制开始界面"""
        # 半透明覆盖层
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # 黑色半透明
        self.screen.blit(overlay, (0, 0))
        
        # 游戏标题（有脉动效果）
        pulse = 1 + 0.1 * np.sin(self.title_pulse)
        self.title_pulse += self.title_pulse_speed
        
        # 使用预先创建的字体，避免动态创建
        # 根据脉动效果选择不同大小的字体
        if pulse > 1.05:
            title_font = self.fonts["title_big"]
        else:
            title_font = self.fonts["title"]
        
        title = title_font.render("Q-FLAPPY BIRD", True, self.COLORS["gold"])
        title_rect = title.get_rect(center=(self.WIDTH//2, 100))
        self.screen.blit(title, title_rect)
        
        # 副标题
        subtitle = self.fonts["subtitle"].render("Reinforcement Learning Demo", True, self.COLORS["white"])
        subtitle_rect = subtitle.get_rect(center=(self.WIDTH//2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        # 游戏说明
        instructions = [
            "HOW TO PLAY:",
            "The AI bird learns to fly through pipes",
            "using Q-learning algorithm.",
            "",
            "GAME CONTROLS:",
            "SPACE - Start game / Restart",
            "ESC   - Exit game",
            "R     - Reset game"
        ]
        
        for i, line in enumerate(instructions):
            color = self.COLORS["gold"] if "HOW" in line or "GAME" in line else self.COLORS["white"]
            text = self.fonts["normal"].render(line, True, color)
            text_rect = text.get_rect(center=(self.WIDTH//2, 220 + i * 25))
            self.screen.blit(text, text_rect)
        
        # 开始提示
        start_y = 450
        if int(pygame.time.get_ticks() / 500) % 2:  # 闪烁效果
            start_text = self.fonts["subtitle"].render("PRESS SPACE TO START", True, self.COLORS["silver"])
            start_rect = start_text.get_rect(center=(self.WIDTH//2, start_y))
            self.screen.blit(start_text, start_rect)
        
        # 最高分
        high_score_text = self.fonts["normal"].render(f"HIGH SCORE: {self.high_score}", True, self.COLORS["bronze"])
        high_score_rect = high_score_text.get_rect(center=(self.WIDTH//2, 500))
        self.screen.blit(high_score_text, high_score_rect)
        
        # 课程信息
        course_text = self.fonts["small"].render("CDS524 Assignment 1 - Reinforcement Learning", True, self.COLORS["white"])
        course_rect = course_text.get_rect(center=(self.WIDTH//2, 550))
        self.screen.blit(course_text, course_rect)

    def _draw_game_over_screen(self):
        """绘制游戏结束界面"""
        # 半透明覆盖层
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # 黑色半透明
        self.screen.blit(overlay, (0, 0))
        
        # 游戏结束文字
        game_over = self.fonts["title"].render("GAME OVER", True, self.COLORS["red"])
        game_over_rect = game_over.get_rect(center=(self.WIDTH//2, 150))
        self.screen.blit(game_over, game_over_rect)
        
        # 最终得分
        score_text = self.fonts["score"].render(f"SCORE: {self.score}", True, self.COLORS["gold"])
        score_rect = score_text.get_rect(center=(self.WIDTH//2, 220))
        self.screen.blit(score_text, score_rect)
        
        # 最高分
        high_score_text = self.fonts["normal"].render(f"HIGH SCORE: {self.high_score}", True, self.COLORS["silver"])
        high_score_rect = high_score_text.get_rect(center=(self.WIDTH//2, 270))
        self.screen.blit(high_score_text, high_score_rect)
        
        # 游戏统计
        stats = [
            f"Pipes passed: {self.score}",
            f"Last action: {'JUMP' if self.last_action == 1 else 'FALL'}",
            f"Last reward: {self.last_reward:.1f}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.fonts["normal"].render(stat, True, self.COLORS["white"])
            text_rect = text.get_rect(center=(self.WIDTH//2, 320 + i * 30))
            self.screen.blit(text, text_rect)
        
        # 重新开始提示
        restart_y = 450
        if int(pygame.time.get_ticks() / 500) % 2:  # 闪烁效果
            restart_text = self.fonts["subtitle"].render("PRESS SPACE TO RESTART", True, self.COLORS["silver"])
            restart_rect = restart_text.get_rect(center=(self.WIDTH//2, restart_y))
            self.screen.blit(restart_text, restart_rect)
        
        # 退出提示
        quit_text = self.fonts["normal"].render("Press ESC to exit", True, self.COLORS["white"])
        quit_rect = quit_text.get_rect(center=(self.WIDTH//2, 500))
        self.screen.blit(quit_text, quit_rect)

    def _draw_hud(self):
        """绘制游戏进行中的HUD（抬头显示器）"""
        # 当前分数
        score_text = self.fonts["score"].render(f"Score: {self.score}", True, self.COLORS["white"])
        self.screen.blit(score_text, (20, 20))
        
        # 最高分
        high_score_text = self.fonts["normal"].render(f"High: {self.high_score}", True, self.COLORS["bronze"])
        self.screen.blit(high_score_text, (20, 60))
        
        # 游戏状态指示器
        status_color = (0, 255, 0) if self.last_reward >= 0 else (255, 0, 0)
        status_text = self.fonts["small"].render(f"Reward: {self.last_reward:.1f}", True, status_color)
        self.screen.blit(status_text, (20, 100))
        
        # 动作指示器
        action_text = self.fonts["small"].render(f"Action: {'JUMP' if self.last_action == 1 else 'FALL'}", 
                                                True, self.COLORS["white"])
        self.screen.blit(action_text, (20, 120))
        
        # 帮助提示
        if self.score < 3:  # 只在开始时显示提示
            help_text = self.fonts["small"].render("AI is learning...", True, (255, 255, 0))
            help_rect = help_text.get_rect(center=(self.WIDTH//2, 50))
            self.screen.blit(help_text, help_rect)

    def _render(self):
        """渲染游戏画面"""
        # 绘制背景
        self._draw_background()
        
        # 绘制管道
        self._draw_pipes()
        
        # 绘制小鸟
        self._draw_bird()
        
        # 根据游戏状态绘制不同的UI
        if self.game_state == "start":
            self._draw_start_screen()
        elif self.game_state == "game_over":
            self._draw_game_over_screen()
        elif self.game_state == "playing":
            self._draw_hud()
        
        # 绘制游戏信息
        info_text = self.fonts["small"].render("Q-Learning AI Demo | Press R to reset", True, self.COLORS["white"])
        info_rect = info_text.get_rect(center=(self.WIDTH//2, self.HEIGHT - 20))
        self.screen.blit(info_text, info_rect)
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

    def close(self):
        """关闭游戏"""
        if self.render_mode == "human":
            pygame.quit()
