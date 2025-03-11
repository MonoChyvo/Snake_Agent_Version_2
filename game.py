"""
Implementación del entorno del juego Snake para Aprendizaje Q Profundo.
Este módulo proporciona la mecánica del juego y visualización para la IA de Snake:

Componentes principales:
- SnakeGameAI: Clase principal del juego con integración de aprendizaje por refuerzo
- Direction: Enumeración para las direcciones de movimiento de la serpiente
- Sistema avanzado de recompensas que incluye:
  * Recompensas basadas en distancia
  * Bonificaciones por supervivencia
  * Penalizaciones por ineficiencia
  * Predicción de colisiones futuras
- Características visuales:
  * Mapa de calor para posiciones visitadas
  * Visualización de puntuación y estadísticas
  * Renderizado de serpiente y comida
- Gestión del estado del juego y detección de colisiones
"""

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Optional, Tuple, List
from config import BLOCK_SIZE, SPEED, BLUE1, BLUE2, RED, WHITE, BLACK
from advanced_pathfinding import AdvancedPathfinding

pygame.init()

try:
    font = pygame.font.Font("arial.ttf", 25)
except FileNotFoundError:
    print("No se encontró 'arial.ttf'. Usando fuente predeterminada.")
    font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SnakeGameAI:
    def __init__(
        self, width: int = 640, height: int = 480, n_game: int = 0, record: int = 0
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.n_game: int = n_game
        self.record: int = record
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Training Snake")
        self.clock = pygame.time.Clock()
        self.pathfinder = AdvancedPathfinding(self)

        # Cargar imagen de manzana
        try:
            self.apple_image = pygame.image.load("apple.png").convert_alpha()
            self.apple_image = pygame.transform.scale(self.apple_image, (BLOCK_SIZE, BLOCK_SIZE))
        except Exception as e:
            print("Error loading apple image, using fallback drawing.")
            self.apple_image = None

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
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score: int = 0
        self.food: Optional[Point] = None
        self._place_food()
        self.frame_iteration: int = 0

        self.reward_history = []
        self.action_history = []
        self.steps = 0

        self.visit_map = np.zeros((self.width // BLOCK_SIZE, self.height // BLOCK_SIZE))
        self.particles = []  # Inicializar lista de partículas para confeti

    def _place_food(self):
        """Coloca comida en una posición aleatoria."""
        while True:
            x: int = (
                random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            )
            y: int = (
                random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            )
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

        # Track food locations for analysis
        if hasattr(self, "food") and self.food is not None:
            self.food_locations.append((self.food.x, self.food.y))

    def play_step(
        self, action: List[int], n_game: int, record: int
    ) -> Tuple[int, bool, int]:
        prev_distance: int = (
            abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if self.food
            else 999999
        )

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

        grid_x, grid_y = self._grid_position(self.head)

        # Add bounds checking before updating visit_map
        grid_width, grid_height = self.grid_size
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            self.visit_map[grid_x, grid_y] += 1

        efficiency_penalty = 0  # Initialize efficiency_penalty to 0
        # Modificar el cálculo de efficiency_penalty
        if len(self.snake) > 5:
            # Add bounds checking before accessing visit_map
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                visit_score = self.visit_map[grid_x, grid_y]
                if visit_score > 2:  # Visitó esta celda más de dos veces
                    efficiency_penalty = -0.05 * (visit_score - 1)
            else:
                # Handle out of bounds case
                efficiency_penalty = 0

        reward: int = 0
        game_over: bool = False

        # Calculate current distance to food after move
        current_distance: int = (
            int(abs(self.head.x - self.food.x) + abs(self.food.y - self.head.y))
            if self.food
            else 999999
        )

        # Check for game over conditions
        if self.is_collision() or self.frame_iteration > 120 * len(self.snake):
            game_over = True
            # Higher penalty for early deaths, less penalty for deaths after longer games
            base_penalty = -10
            survival_factor = min(
                len(self.snake) // 10, 1
            )  # Cap at 1, using integer division
            reward = int(base_penalty * (1 - 0.5 * survival_factor))  # Convert to int
            self.reward_history.append(reward)
            return reward, game_over, self.score

        # Check for food eaten
        if self.head == self.food:
            self.score += 1
            # Generar partículas de confeti al comer comida
            self.spawn_confetti(self.food)
            # Reward scales with snake length - more reward for growing longer
            base_reward = 1.0
            length_bonus = min(len(self.snake) * 0.5, 10)  # Cap at +10 bonus
            reward = int(base_reward + length_bonus)
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
                    future_penalty = -0.1 * (
                        4 - look_ahead
                    )  # -0.3 para 1 paso, -0.2 para 2 pasos, -0.1 para 3 pasos
                    break

            # Combined reward - añadir future_penalty a la combinación
            reward = int(
                distance_reward
                + survival_reward
                + efficiency_penalty
                + danger_reward
                + future_penalty
            )

            # Store reward and remove tail
            self.reward_history.append(reward)
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def spawn_confetti(self, position):
        """Genera partículas de confeti en la posición dada."""
        num_particles = 20
        for _ in range(num_particles):
            particle = {
                'pos': [position.x + BLOCK_SIZE // 2, position.y + BLOCK_SIZE // 2],
                'vel': [random.uniform(-2, 2), random.uniform(-2, 2)],
                'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                'lifetime': random.uniform(20, 40)
            }
            self.particles.append(particle)

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
        if (
            point.x >= self.width
            or point.x < 0
            or point.y >= self.height
            or point.y < 0
        ):
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """Actualiza la interfaz gráfica."""
        self.display.fill(BLACK)

        # Actualizar y renderizar partículas (efecto confeti)
        for particle in self.particles[:]:
            # Actualizar posición y disminuir lifetime
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['lifetime'] -= 1
            # Dibujar partícula
            pygame.draw.circle(self.display, particle['color'], (int(particle['pos'][0]), int(particle['pos'][1])), 2)
            if particle['lifetime'] <= 0:
                self.particles.remove(particle)

        # Renderizado mejorado de la serpiente con sombra
        for i, pt in enumerate(self.snake):
            # Definir el rectángulo del segmento
            snake_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            # Dibujar sombra (desplazada 3 píxeles a la derecha y abajo)
            shadow_offset = 3
            shadow_color = (50, 50, 50)
            shadow_rect = snake_rect.copy()
            shadow_rect.x += shadow_offset
            shadow_rect.y += shadow_offset
            pygame.draw.rect(
                self.display,
                shadow_color,
                shadow_rect,
                border_radius=BLOCK_SIZE // 2
            )
            # Dibujar el segmento de la serpiente
            color_factor = 1 - (i / len(self.snake))
            body_color = (
                int(30 + 225 * color_factor),
                int(144 - 39 * color_factor),
                int(255 - 75 * color_factor),
            )
            pygame.draw.rect(
                self.display,
                body_color,
                snake_rect,
                border_radius=BLOCK_SIZE // 2
            )

        # Comida: usar imagen PNG de manzana si está disponible
        if self.food is not None:
            apple_rect = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE * 1.5, BLOCK_SIZE * 1.5)  # Increase size by 1.5x
            if self.apple_image:
                self.apple_image = pygame.transform.scale(self.apple_image, (BLOCK_SIZE, BLOCK_SIZE))  # Resize image
                self.display.blit(self.apple_image, apple_rect)
            else:
                pygame.draw.rect(self.display, (255, 0, 0), apple_rect, border_radius=5)
                pygame.draw.rect(self.display, (0, 0, 0), apple_rect, 2, border_radius=5)

        score_text = font.render(f"Score: {self.score}", True, WHITE)
        n_game_text = font.render(f"Game: {self.n_game + 1}", True, WHITE)
        record_text = font.render(f"Record: {self.record}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        self.display.blit(n_game_text, [0, 30])
        self.display.blit(record_text, [0, 60])

        # Añadir indicador visual para el estado de pathfinding
        if "agent" in globals() and hasattr(globals()["agent"], "pathfinding_enabled"):
            pathfinding_status = (
                "ON" if globals()["agent"].pathfinding_enabled else "OFF"
            )
            pathfinding_color = (
                (0, 255, 0) if globals()["agent"].pathfinding_enabled else (255, 0, 0)
            )
            pathfinding_text = font.render(
                f"Pathfinding: {pathfinding_status}", True, pathfinding_color
            )
            self.display.blit(pathfinding_text, [self.width - 200, 0])

            # Añadir instrucción para el usuario
            help_font = (
                pygame.font.Font("arial.ttf", 15)
                if pygame.font.get_init()
                else pygame.font.SysFont("arial", 15)
            )
            help_text = help_font.render("Press 'P' to toggle pathfinding", True, WHITE)
            self.display.blit(help_text, [self.width - 200, 30])

        pygame.display.flip()

    def _move(self, action: List[int]) -> None:
        """
        Mueve la serpiente según la acción dada.
        Acción: [recto, derecha, izquierda]
        """
        directions: List[Direction] = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
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

    def _grid_position(self, point):
        return (point.x // BLOCK_SIZE, point.y // BLOCK_SIZE)

    @property
    def grid_size(self):
        return (self.width // BLOCK_SIZE, self.height // BLOCK_SIZE)

    def find_path(self):
        return self.pathfinder.find_optimal_path()

    def _safe_moves(self, path):
        if len(self.snake) < 15:
            return path  # Aggressive mode for short snakes

        # Defensive mode: Prefer paths with more escape routes
        safe_path = []
        for pos in path:
            x, y = pos
            neighbors = [
                (x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
            open_neighbors = sum(
                1
                for n in neighbors
                if 0 <= n[0] < self.grid_size[0]
                and 0 <= n[1] < self.grid_size[1]
                and not any(
                    p.x // BLOCK_SIZE == n[0] and p.y // BLOCK_SIZE == n[1]
                    for p in self.snake
                )
            )
            if open_neighbors >= 2:
                safe_path.append(pos)
        return safe_path or path
