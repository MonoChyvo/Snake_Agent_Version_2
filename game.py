import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Optional, Tuple, List
from config import *

pygame.init()

try:
    font = pygame.font.Font('arial.ttf', 25)
except FileNotFoundError:
    print("No se encontró 'arial.ttf'. Usando fuente predeterminada.")
    font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, width: int = 640, height: int = 480, n_game: int = 0, record: int = 0) -> None:
        self.width: int = width
        self.height: int = height
        self.n_game: int = n_game
        self.record: int = record
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Training Snake')
        self.clock = pygame.time.Clock()
        
        self.steps = 0
        self.reward_history = []
        self.action_history = []
        self.food_locations = []
        
        self.reset()

    def reset(self):
        """Reinicia el estado del juego."""
        self.direction: Direction = random.choice(list(Direction))
        self.head: Point = Point(self.width // 2, self.height // 2)
        self.snake: List[Point] = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score: int = 0
        self.food: Optional[Point] = None
        self._place_food()
        self.frame_iteration: int = 0
        
        self.reward_history = []
        self.action_history = []
        self.steps = 0
        
        self.visit_map = np.zeros((self.width // BLOCK_SIZE, self.height // BLOCK_SIZE))

    def _place_food(self):
        """Coloca comida en una posición aleatoria."""
        while True:
            x: int = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y: int = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
            
        # Track food locations for analysis
        if hasattr(self, "food") and self.food is not None:
            self.food_locations.append((self.food.x, self.food.y))

    def play_step(self, action: List[int], n_game: int, record: int) -> Tuple[int, bool, int]:
        prev_distance: int = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        action_idx = np.argmax(action) if isinstance(action, list) else action
        self.action_history.append(action_idx)
    
        self.steps += 1
        self.n_game = n_game
        self.record = record
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)
        
        grid_x, grid_y = self.head.x // BLOCK_SIZE, self.head.y // BLOCK_SIZE
        # Ensure grid_x and grid_y are within bounds
        grid_x = min(max(grid_x, 0), self.visit_map.shape[0] - 1)
        grid_y = min(max(grid_y, 0), self.visit_map.shape[1] - 1)
        self.visit_map[grid_x, grid_y] += 1
        
        efficiency_penalty = 0  # Initialize efficiency_penalty to 0
        # Modificar el cálculo de efficiency_penalty
        if len(self.snake) > 5:
            visit_score = self.visit_map[grid_x, grid_y]
            if visit_score > 2:  # Visitó esta celda más de dos veces
                efficiency_penalty = -0.05 * (visit_score - 1)
            else:
                efficiency_penalty = 0

        reward: int = 0
        game_over: bool = False
        ate_food: bool = False
        
        # Calculate current distance to food after move
        current_distance: int = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    
        
        # Check for game over conditions
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            # Higher penalty for early deaths, less penalty for deaths after longer games
            base_penalty = -10
            survival_factor = min(len(self.snake) / 10, 1.0)  # Cap at 1.0
            reward = base_penalty * (1 - 0.5 * survival_factor)  # Less penalty if snake is longer
            self.reward_history.append(reward)
            return reward, game_over, self.score, ate_food

        # Check for food eaten
        if self.head == self.food:
            self.score += 1
            ate_food = True
            # Reward scales with snake length - more reward for growing longer
            base_reward = 1.0
            length_bonus = min(len(self.snake) * 0.5, 10)  # Cap at +10 bonus
            reward = base_reward + length_bonus
            self.reward_history.append(reward)
            self._place_food()
        else:
            # Mejora la recompensa basada en distancia en el método play_step
            # Distance-based reward component con enfoque progresivo
            distance_change = prev_distance - current_distance
            # Usar una función no lineal para premiar más los acercamientos significativos
            if distance_change > 0:  # Se está acercando a la comida
                # Recompensa mayor por acercamientos grandes
                distance_reward = 0.01 * (1 + distance_change / 10)
            elif distance_change < 0:  # Se está alejando de la comida
                # Penalizar menos por pequeños alejamientos
                distance_reward = 0.01 * distance_change / 2
            else:
                distance_reward = 0

            # Survival reward - small bonus for staying alive
            survival_reward = 0.001

            # Danger awareness reward - more space is better
            danger_reward = 0
            # Check if next move in current direction would be safe
            next_x, next_y = self._get_next_position(self.head, self.direction)
            next_pos = Point(next_x, next_y)
            if not self.is_collision(next_pos):
                danger_reward += 0.01
                
            # Verificar trayectoria futura para detectar colisiones inminentes
            future_penalty = 0
            future_pos = self.head
            future_dir = self.direction

            # Simular hasta 3 pasos adelante en la dirección actual
            for look_ahead in range(1, 4):
                next_x, next_y = self._get_next_position(future_pos, future_dir)
                future_pos = Point(next_x, next_y)
                
                # Si vamos a chocar en los próximos 3 movimientos, penalizar proporcionalmente
                if self.is_collision(future_pos):
                    future_penalty = -0.1 * (4 - look_ahead)  # -0.3 para 1 paso, -0.2 para 2 pasos, -0.1 para 3 pasos
                    break

            # Combined reward - añadir future_penalty a la combinación
            reward = distance_reward + survival_reward + efficiency_penalty + danger_reward + future_penalty
            
            

            # Store reward and remove tail
            self.reward_history.append(reward)
            self.snake.pop()
            
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score, ate_food
    
    def _get_next_position(self, point, direction):
        """Helper to get next position in a given direction."""
        x, y = point.x, point.y
        
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        return x, y

    def is_collision(self, point: Optional[Point] = None) -> bool:
        """Verifica si hay colisión."""
        if point is None:
            point = self.head
        if (point.x >= self.width or point.x < 0 or point.y >= self.height or point.y < 0):
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _update_ui(self) -> None:
        """Actualiza la interfaz gráfica."""
        self.display.fill(BLACK)
        
        if hasattr(self, 'visit_map'):
            max_visits = np.max(self.visit_map) if np.max(self.visit_map) > 0 else 1
            for x in range(self.width // BLOCK_SIZE):
                for y in range(self.height // BLOCK_SIZE):
                    visits = self.visit_map[x, y]
                    if visits > 0:
                        # Color intensity based on visit count (darker = more visits)
                        intensity = min(255, int(100 + (visits / max_visits) * 155))
                        heat_color = (intensity, 0, 0)  # Rojo con intensidad variable
                        pygame.draw.rect(
                            self.display, 
                            heat_color, 
                            pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                            1  # Solo borde para no ocultar la serpiente
                        )
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
    
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        n_game_text = font.render(f"Game: {self.n_game}", True, WHITE)
        record_text = font.render(f"Record: {self.record}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        self.display.blit(n_game_text, [0, 30])
        self.display.blit(record_text, [0, 60])
        pygame.display.flip()

    def _move(self, action: List[int]) -> None:
        """
        Mueve la serpiente según la acción dada.
        Acción: [recto, derecha, izquierda]
        """
        directions: List[Direction] = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx: int = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir: Direction = directions[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir: Direction = directions[(idx + 1) % 4]
        else:
            new_dir: Direction = directions[(idx - 1) % 4]

        self.direction = new_dir

        x: int = self.head.x
        y: int = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)