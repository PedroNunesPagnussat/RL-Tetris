import pygame
import random
import numpy as np

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (100, 100, 100)
GRID_COLOR = (50, 50, 50)
BACKGROUND_COLOR = (20, 20, 20)

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]]   # Z
]

SHAPE_COLORS = [CYAN, YELLOW, MAGENTA, ORANGE, BLUE, GREEN, RED]

class TetrisGame:
    # Game dimensions
    BLOCK_SIZE = 30
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    SCREEN_WIDTH = BLOCK_SIZE * (GRID_WIDTH + 8)
    SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT
    PADDING = 5

    def __init__(self, render=False):
        self.render = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption('Tetris')
            self.font = pygame.font.Font(None, 36)
            self.big_font = pygame.font.Font(None, 48)
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.hold_piece = None
        self.can_hold = True
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        return self.get_state()

    def new_piece(self):
        shape_idx = random.randint(0, len(SHAPES) - 1)
        return {
            'shape': SHAPES[shape_idx],
            'color': SHAPE_COLORS[shape_idx],
            'x': self.GRID_WIDTH // 2 - len(SHAPES[shape_idx][0]) // 2,
            'y': 0
        }

    def valid_move(self, piece, x, y):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    new_x = x + j
                    new_y = y + i
                    if (new_x < 0 or new_x >= self.GRID_WIDTH or 
                        new_y >= self.GRID_HEIGHT or 
                        (new_y >= 0 and self.grid[new_y][new_x])):
                        return False
        return True

    def rotate_piece(self, piece):
        return {
            'shape': list(zip(*piece['shape'][::-1])),
            'color': piece['color'],
            'x': piece['x'],
            'y': piece['y']
        }

    def lock_piece(self, piece):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.grid[piece['y'] + i][piece['x'] + j] = piece['color']
        self.clear_lines()

    def clear_lines(self):
        lines_cleared = 0
        for i in range(self.GRID_HEIGHT):
            if all(self.grid[i]):
                del self.grid[i]
                self.grid.insert(0, [0 for _ in range(self.GRID_WIDTH)])
                lines_cleared += 1
        
        self.lines_cleared += lines_cleared
        self.level = (self.lines_cleared // 10) + 1
        
        if lines_cleared > 0:
            base_scores = {1: 100, 2: 300, 3: 500, 4: 800}
            self.score += base_scores[lines_cleared] * self.level
            return lines_cleared
        return 0

    def get_state(self):
        # Convert grid to numpy array for RL
        state = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x]:
                    state[y][x] = 1
        
        # Add current piece to state
        piece_state = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32)
        for i, row in enumerate(self.current_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    y = self.current_piece['y'] + i
                    x = self.current_piece['x'] + j
                    if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                        piece_state[y][x] = 1
        
        return np.stack([state, piece_state], axis=0)

    def count_holes(self):
        holes = 0
        for x in range(self.GRID_WIDTH):
            found_block = False
            for y in range(self.GRID_HEIGHT):
                if self.grid[y][x]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def get_column_heights(self):
        heights = []
        for x in range(self.GRID_WIDTH):
            height = 0
            for y in range(self.GRID_HEIGHT):
                if self.grid[y][x]:
                    height = self.GRID_HEIGHT - y
                    break
            heights.append(height)
        return heights

    def get_bumpiness(self):
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def compute_reward(self, prev_lines, prev_holes, prev_max_height):
        """Compute the reward based on game state changes."""
        lines_cleared_now = self.lines_cleared - prev_lines
        holes_now = self.count_holes()
        max_height_now = max(self.get_column_heights())
        bumpiness = self.get_bumpiness()

        # === Reward weights ===
        line_reward_table = {1: 100, 2: 300, 3: 500, 4: 800}
        hole_penalty_weight = 5
        height_penalty_weight = 10
        bumpiness_penalty_weight = 1
        survival_bonus = 1

        reward = 0.0
        reward += line_reward_table.get(lines_cleared_now, 0.0)
        reward -= hole_penalty_weight * (holes_now - prev_holes)
        reward -= height_penalty_weight * (max_height_now - prev_max_height)
        reward -= bumpiness_penalty_weight * bumpiness
        reward += survival_bonus

        # Normalize reward to stabilize training
        reward /= 100.0

        return reward


    def step(self, action):

        prev_holes = self.count_holes()
        prev_max_height = max(self.get_column_heights())
        prev_lines = self.lines_cleared

        # === Execute action ===
        if action == 0:  # Left
            if self.valid_move(self.current_piece, self.current_piece['x'] - 1, self.current_piece['y']):
                self.current_piece['x'] -= 1

        elif action == 1:  # Right
            if self.valid_move(self.current_piece, self.current_piece['x'] + 1, self.current_piece['y']):
                self.current_piece['x'] += 1

        elif action == 2:  # Rotate
            rotated = self.rotate_piece(self.current_piece)
            if self.valid_move(rotated, rotated['x'], rotated['y']):
                self.current_piece = rotated

        elif action == 3:  # Down
            if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                self.current_piece['y'] += 1

        elif action == 4:  # Drop
            while self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                self.current_piece['y'] += 1
                
        elif action == 5:  # Hold
            self.hold_current_piece()

        # === Gravity (1 step down) ===
        if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
            self.current_piece['y'] += 1
            done = False

        else:
            self.lock_piece(self.current_piece)
            self.current_piece = self.next_piece
            self.next_piece = self.new_piece()
            self.can_hold = True

            if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
                return self.get_state(), -1.0, True  # Game Over

        reward = self.compute_reward(prev_lines, prev_holes, prev_max_height)
        return self.get_state(), reward, False
    

    def render_game(self):
        if not self.render:
            return

        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_piece(self.current_piece, is_current=True)
        self.draw_score()
        self.draw_next_piece()
        self.draw_hold_piece()
        pygame.display.flip()
        self.clock.tick(60)

    def draw_block(self, x, y, color, is_current=False):
        if not self.render:
            return

        shadow_offset = 2
        pygame.draw.rect(self.screen, (0, 0, 0),
                        (x * self.BLOCK_SIZE + shadow_offset,
                         y * self.BLOCK_SIZE + shadow_offset,
                         self.BLOCK_SIZE - 2, self.BLOCK_SIZE - 2))
        
        pygame.draw.rect(self.screen, color,
                        (x * self.BLOCK_SIZE, y * self.BLOCK_SIZE,
                         self.BLOCK_SIZE - 2, self.BLOCK_SIZE - 2))
        
        if is_current:
            highlight_color = (min(color[0] + 50, 255),
                             min(color[1] + 50, 255),
                             min(color[2] + 50, 255))
            pygame.draw.rect(self.screen, highlight_color,
                           (x * self.BLOCK_SIZE + 2, y * self.BLOCK_SIZE + 2,
                            self.BLOCK_SIZE - 6, self.BLOCK_SIZE - 6))

    def draw_grid(self):
        if not self.render:
            return

        pygame.draw.rect(self.screen, BACKGROUND_COLOR,
                        (0, 0, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                pygame.draw.rect(self.screen, GRID_COLOR,
                               (x * self.BLOCK_SIZE, y * self.BLOCK_SIZE,
                                self.BLOCK_SIZE, self.BLOCK_SIZE), 1)
                if self.grid[y][x]:
                    self.draw_block(x, y, self.grid[y][x])

    def draw_piece(self, piece, is_current=False):
        if not self.render:
            return

        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.draw_block(piece['x'] + j, piece['y'] + i,
                                  piece['color'], is_current)

    def draw_score(self):
        if not self.render:
            return

        panel_x = self.GRID_WIDTH * self.BLOCK_SIZE + self.PADDING
        panel_width = self.SCREEN_WIDTH - panel_x - self.PADDING
        pygame.draw.rect(self.screen, DARK_GRAY,
                        (panel_x, 0, panel_width, self.SCREEN_HEIGHT))
        
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        level_text = self.font.render(f'Level: {self.level}', True, WHITE)
        lines_text = self.font.render(f'Lines: {self.lines_cleared}', True, WHITE)
        
        self.screen.blit(score_text, (panel_x + 10, 20))
        self.screen.blit(level_text, (panel_x + 10, 60))
        self.screen.blit(lines_text, (panel_x + 10, 100))

    def draw_next_piece(self):
        if not self.render:
            return

        panel_x = self.GRID_WIDTH * self.BLOCK_SIZE + self.PADDING
        next_text = self.font.render('Next:', True, WHITE)
        self.screen.blit(next_text, (panel_x + 10, 150))
        
        preview_x = panel_x + 30
        preview_y = 200
        for i, row in enumerate(self.next_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.draw_block((preview_x + j * self.BLOCK_SIZE) // self.BLOCK_SIZE,
                                  (preview_y + i * self.BLOCK_SIZE) // self.BLOCK_SIZE,
                                  self.next_piece['color'])

    def draw_hold_piece(self):
        if not self.render:
            return

        panel_x = self.GRID_WIDTH * self.BLOCK_SIZE + self.PADDING
        hold_text = self.font.render('Hold:', True, WHITE)
        self.screen.blit(hold_text, (panel_x + 10, 300))
        
        if self.hold_piece:
            preview_x = panel_x + 30
            preview_y = 350
            for i, row in enumerate(self.hold_piece['shape']):
                for j, cell in enumerate(row):
                    if cell:
                        self.draw_block((preview_x + j * self.BLOCK_SIZE) // self.BLOCK_SIZE,
                                      (preview_y + i * self.BLOCK_SIZE) // self.BLOCK_SIZE,
                                      self.hold_piece['color'])

    def hold_current_piece(self):
        if not self.can_hold:
            return
            
        if self.hold_piece is None:
            self.hold_piece = self.current_piece
            self.current_piece = self.next_piece
            self.next_piece = self.new_piece()
        
        else:
            temp = self.current_piece
            self.current_piece = self.hold_piece
            self.hold_piece = temp
            self.current_piece['x'] = self.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
            self.current_piece['y'] = 0
            
        self.can_hold = False