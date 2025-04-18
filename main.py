import pygame
import random

# Initialize Pygame
pygame.init()

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

# Game dimensions
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * (GRID_WIDTH + 8)  # Extra space for next piece
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT
PADDING = 5

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

class Tetris:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()
        self.reset_game()
        
        # Load font
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 48)

    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.hold_piece = None
        self.can_hold = True  # Flag to prevent multiple holds per piece
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0

    def new_piece(self):
        # Choose a random shape
        shape_idx = random.randint(0, len(SHAPES) - 1)
        # Create piece with position and rotation
        return {
            'shape': SHAPES[shape_idx],
            'color': SHAPE_COLORS[shape_idx],
            'x': GRID_WIDTH // 2 - len(SHAPES[shape_idx][0]) // 2,
            'y': 0
        }

    def valid_move(self, piece, x, y):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    new_x = x + j
                    new_y = y + i
                    if (new_x < 0 or new_x >= GRID_WIDTH or 
                        new_y >= GRID_HEIGHT or 
                        (new_y >= 0 and self.grid[new_y][new_x])):
                        return False
        return True

    def rotate_piece(self, piece):
        # Transpose and reverse rows to rotate 90 degrees clockwise
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
        # Check for completed lines
        self.clear_lines()

    def hold_current_piece(self):
        if not self.can_hold:
            return
            
        if self.hold_piece is None:
            self.hold_piece = self.current_piece
            self.current_piece = self.next_piece
            self.next_piece = self.new_piece()
        else:
            # Swap current piece with hold piece
            temp = self.current_piece
            self.current_piece = self.hold_piece
            self.hold_piece = temp
            # Reset position of the swapped piece
            self.current_piece['x'] = GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
            self.current_piece['y'] = 0
            
        self.can_hold = False

    def clear_lines(self):
        lines_cleared = 0
        for i in range(GRID_HEIGHT):
            if all(self.grid[i]):
                del self.grid[i]
                self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
                lines_cleared += 1
        
        # Update total lines cleared and level
        self.lines_cleared += lines_cleared
        self.level = (self.lines_cleared // 10) + 1  # Level up every 10 lines
        
        # Score calculation based on standard Tetris scoring
        if lines_cleared > 0:
            base_scores = {1: 100, 2: 300, 3: 500, 4: 800}
            self.score += base_scores[lines_cleared] * self.level

    def draw_block(self, x, y, color, is_current=False):
        # Draw block shadow
        shadow_offset = 2
        pygame.draw.rect(self.screen, (0, 0, 0),
                        (x * BLOCK_SIZE + shadow_offset,
                         y * BLOCK_SIZE + shadow_offset,
                         BLOCK_SIZE - 2, BLOCK_SIZE - 2))
        
        # Draw main block
        pygame.draw.rect(self.screen, color,
                        (x * BLOCK_SIZE, y * BLOCK_SIZE,
                         BLOCK_SIZE - 2, BLOCK_SIZE - 2))
        
        # Draw highlight for current piece
        if is_current:
            highlight_color = (min(color[0] + 50, 255),
                             min(color[1] + 50, 255),
                             min(color[2] + 50, 255))
            pygame.draw.rect(self.screen, highlight_color,
                           (x * BLOCK_SIZE + 2, y * BLOCK_SIZE + 2,
                            BLOCK_SIZE - 6, BLOCK_SIZE - 6))

    def draw_grid(self):
        # Draw background
        pygame.draw.rect(self.screen, BACKGROUND_COLOR,
                        (0, 0, GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE))
        
        # Draw grid lines
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pygame.draw.rect(self.screen, GRID_COLOR,
                               (x * BLOCK_SIZE, y * BLOCK_SIZE,
                                BLOCK_SIZE, BLOCK_SIZE), 1)
                if self.grid[y][x]:
                    self.draw_block(x, y, self.grid[y][x])

    def draw_piece(self, piece, is_current=False):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.draw_block(piece['x'] + j, piece['y'] + i,
                                  piece['color'], is_current)

    def draw_score(self):
        # Draw score panel background
        panel_x = GRID_WIDTH * BLOCK_SIZE + PADDING
        panel_width = SCREEN_WIDTH - panel_x - PADDING
        pygame.draw.rect(self.screen, DARK_GRAY,
                        (panel_x, 0, panel_width, SCREEN_HEIGHT))
        
        # Draw score elements
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        level_text = self.font.render(f'Level: {self.level}', True, WHITE)
        lines_text = self.font.render(f'Lines: {self.lines_cleared}', True, WHITE)
        
        self.screen.blit(score_text, (panel_x + 10, 20))
        self.screen.blit(level_text, (panel_x + 10, 60))
        self.screen.blit(lines_text, (panel_x + 10, 100))

    def draw_next_piece(self):
        # Draw next piece panel
        panel_x = GRID_WIDTH * BLOCK_SIZE + PADDING
        next_text = self.font.render('Next:', True, WHITE)
        self.screen.blit(next_text, (panel_x + 10, 150))
        
        # Draw the next piece preview
        preview_x = panel_x + 30
        preview_y = 200
        for i, row in enumerate(self.next_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.draw_block((preview_x + j * BLOCK_SIZE) // BLOCK_SIZE,
                                  (preview_y + i * BLOCK_SIZE) // BLOCK_SIZE,
                                  self.next_piece['color'])

    def draw_hold_piece(self):
        # Draw hold piece panel
        panel_x = GRID_WIDTH * BLOCK_SIZE + PADDING
        hold_text = self.font.render('Hold:', True, WHITE)
        self.screen.blit(hold_text, (panel_x + 10, 300))
        
        # Draw the hold piece preview if it exists
        if self.hold_piece:
            preview_x = panel_x + 30
            preview_y = 350
            for i, row in enumerate(self.hold_piece['shape']):
                for j, cell in enumerate(row):
                    if cell:
                        self.draw_block((preview_x + j * BLOCK_SIZE) // BLOCK_SIZE,
                                      (preview_y + i * BLOCK_SIZE) // BLOCK_SIZE,
                                      self.hold_piece['color'])

    def run(self):
        fall_time = 0
        fall_speed = 0.5  # Initial fall speed in seconds
        while not self.game_over:
            fall_time += self.clock.get_rawtime()
            self.clock.tick()

            # Calculate current fall speed based on level
            current_fall_speed = max(0.05, fall_speed - (self.level - 1) * 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        if self.valid_move(self.current_piece, self.current_piece['x'] - 1, self.current_piece['y']):
                            self.current_piece['x'] -= 1
                    elif event.key == pygame.K_RIGHT:
                        if self.valid_move(self.current_piece, self.current_piece['x'] + 1, self.current_piece['y']):
                            self.current_piece['x'] += 1
                    elif event.key == pygame.K_DOWN:
                        if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                            self.current_piece['y'] += 1
                    elif event.key == pygame.K_UP:
                        rotated = self.rotate_piece(self.current_piece)
                        if self.valid_move(rotated, rotated['x'], rotated['y']):
                            self.current_piece = rotated
                    elif event.key == pygame.K_SPACE:
                        while self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                            self.current_piece['y'] += 1
                    elif event.key == pygame.K_c:
                        self.hold_current_piece()

            if fall_time >= current_fall_speed * 1000:
                if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
                    self.current_piece['y'] += 1
                else:
                    self.lock_piece(self.current_piece)
                    self.current_piece = self.next_piece
                    self.next_piece = self.new_piece()
                    self.can_hold = True
                    if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
                        self.game_over = True
                fall_time = 0

            self.screen.fill(BACKGROUND_COLOR)
            self.draw_grid()
            self.draw_piece(self.current_piece, is_current=True)
            self.draw_score()
            self.draw_next_piece()
            self.draw_hold_piece()
            pygame.display.flip()

        # Game over screen
        game_over_text = self.big_font.render('GAME OVER', True, WHITE)
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(game_over_text, text_rect)
        pygame.display.flip()
        pygame.time.wait(2000)

if __name__ == "__main__":
    game = Tetris()
    game.run()
    pygame.quit()