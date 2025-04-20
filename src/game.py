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
import os
import logging
from enum import Enum
from collections import namedtuple
from typing import Optional, Tuple, List, Dict, Any, Union
from utils.config import (
    BLOCK_SIZE, SPEED, BLUE1, BLUE2, RED, WHITE, BLACK, GREEN, YELLOW, GRAY, PURPLE,
    VISUAL_MODE, SHOW_GRID, SHOW_HEATMAP, PARTICLE_EFFECTS, SHADOW_EFFECTS, ANIMATION_SPEED,
    HEATMAP_OPACITY, STADIUM_MARGIN_TOP, STADIUM_MARGIN_SIDE, STADIUM_MARGIN_BOTTOM
)
from utils.advanced_pathfinding import AdvancedPathfinding

# Importar funciones de validación si están disponibles
try:
    from utils.validation import validate_resource_file
    validation_available = True
except ImportError:
    validation_available = False
    logging.warning("Módulo de validación no disponible. Se omitirá la validación de recursos.")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("game.log"),
        logging.StreamHandler()
    ]
)

pygame.init()

# Función para cargar recursos con validación
def load_resource(resource_path: str, resource_type: str = "font", size: int = 25) -> Union[pygame.font.Font, pygame.Surface, None]:
    """
    Carga un recurso (fuente, imagen) con validación y manejo de excepciones.

    Args:
        resource_path: Ruta al recurso
        resource_type: Tipo de recurso ('font' o 'image')
        size: Tamaño para fuentes o factor de escala para imágenes

    Returns:
        El recurso cargado o None si ocurre un error
    """
    try:
        # Validar que el archivo existe
        if not os.path.exists(resource_path):
            error_msg = f"El archivo de recurso no existe: {resource_path}"
            logging.error(error_msg)
            return None

        # Validar el recurso si está disponible la validación
        if validation_available:
            try:
                allowed_extensions = ['.ttf', '.otf'] if resource_type == "font" else ['.png', '.jpg', '.jpeg']
                validate_resource_file(resource_path, allowed_extensions)
            except Exception as e:
                logging.error(f"Error de validación del recurso {resource_path}: {e}")
                raise

        # Cargar el recurso según su tipo
        if resource_type == "font":
            return pygame.font.Font(resource_path, size)
        elif resource_type == "image":
            image = pygame.image.load(resource_path).convert_alpha()
            if size != 1:  # Si se especifica un tamaño diferente de 1, escalar la imagen
                image = pygame.transform.scale(image, (size, size))
            return image
        else:
            error_msg = f"Tipo de recurso no soportado: {resource_type}"
            logging.error(error_msg)
            return None

    except FileNotFoundError:
        error_msg = f"No se encontró el archivo: {resource_path}"
        logging.error(error_msg)
        return None
    except pygame.error as e:
        error_msg = f"Error de Pygame al cargar {resource_path}: {e}"
        logging.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error inesperado al cargar {resource_path}: {e}"
        logging.error(error_msg)
        return None

# Cargar fuente principal con manejo de errores
font = load_resource("assets/arial.ttf", "font", 25)
if font is None:
    logging.warning("No se pudo cargar 'assets/arial.ttf'. Usando fuente predeterminada.")
    font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SnakeGameAI:
    def __init__(
        self, width: int = 900, height: int = 700, n_game: int = 0, record: int = 0,
        visual_config: Optional[Dict[str, Any]] = None
    ) -> None:
        # Calcular el tamaño de la cuadrícula de juego (sin márgenes)
        self.grid_width_blocks = (width - 2 * STADIUM_MARGIN_SIDE) // BLOCK_SIZE
        self.grid_height_blocks = (height - STADIUM_MARGIN_TOP - STADIUM_MARGIN_BOTTOM) // BLOCK_SIZE

        # Recalcular el tamaño total de la ventana para que se ajuste perfectamente
        self.width: int = self.grid_width_blocks * BLOCK_SIZE + 2 * STADIUM_MARGIN_SIDE
        self.height: int = self.grid_height_blocks * BLOCK_SIZE + STADIUM_MARGIN_TOP + STADIUM_MARGIN_BOTTOM

        self.n_game: int = n_game
        self.record: int = record
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Training Snake")
        self.clock = pygame.time.Clock()
        self.pathfinder = AdvancedPathfinding(self)

        # Guardar los márgenes para usar en el renderizado
        self.margin_top = STADIUM_MARGIN_TOP
        self.margin_side = STADIUM_MARGIN_SIDE
        self.margin_bottom = STADIUM_MARGIN_BOTTOM

        # Configuración visual
        self.visual_config = visual_config or {
            "visual_mode": VISUAL_MODE,
            "show_grid": SHOW_GRID,
            "show_heatmap": SHOW_HEATMAP,
            "particle_effects": PARTICLE_EFFECTS,
            "shadow_effects": SHADOW_EFFECTS
        }

        # Cargar fuentes con validación
        self.main_font = load_resource("assets/arial.ttf", "font", 25)
        if self.main_font is None:
            logging.warning("No se pudo cargar la fuente principal. Usando fuente predeterminada.")
            self.main_font = pygame.font.SysFont("arial", 25)

        self.small_font = load_resource("assets/arial.ttf", "font", 15)
        if self.small_font is None:
            logging.warning("No se pudo cargar la fuente pequeña. Usando fuente predeterminada.")
            self.small_font = pygame.font.SysFont("arial", 15)

        # Cargar imagen de manzana con validación
        self.apple_image = load_resource("assets/apple.png", "image", BLOCK_SIZE)
        if self.apple_image is None:
            logging.warning("No se pudo cargar la imagen de la manzana. Se usará un dibujo alternativo.")

        # Crear superficie para el mapa de calor
        self.heatmap_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Inicializar variables de seguimiento
        self.steps = 0
        self.reward_history = []
        self.action_history = []
        self.food_locations = []
        self.food_distances = []
        self.decision_times = []
        self.game_duration = 0
        self.avg_open_space_ratio = 0

        self.reset()

    def reset(self):
        """Reinicia el estado del juego manteniendo la configuración visual.
        Inicializa la serpiente en una posición segura con validación mejorada de límites."""
        # Guardar configuración visual actual y dimensiones de la ventana
        current_visual_config = self.visual_config.copy()
        current_width = self.width
        current_height = self.height

        # Usar el tamaño de la cuadrícula calculado en __init__
        grid_width, grid_height = self.grid_width_blocks, self.grid_height_blocks

        # Definir una zona segura para la inicialización (evitar los bordes)
        safe_margin = 5  # Bloques de margen desde el borde

        # Asegurarse de que el margen no sea mayor que la mitad del grid
        safe_margin = min(safe_margin, grid_width // 4, grid_height // 4)

        # Calcular límites seguros para la posición inicial
        min_x, max_x = safe_margin, grid_width - safe_margin - 1
        min_y, max_y = safe_margin, grid_height - safe_margin - 1

        # Verificar que haya espacio suficiente para la serpiente
        if min_x >= max_x or min_y >= max_y:
            # Si el grid es muy pequeño, usar todo el espacio disponible
            min_x, max_x = 1, grid_width - 2
            min_y, max_y = 1, grid_height - 2

        # Elegir una posición aleatoria dentro de la zona segura
        grid_x = random.randint(min_x, max_x)
        grid_y = random.randint(min_y, max_y)

        # Convertir a coordenadas de píxeles
        x = grid_x * BLOCK_SIZE
        y = grid_y * BLOCK_SIZE

        # Elegir una dirección que garantice que la serpiente quepa en el grid
        # Verificar qué direcciones son seguras (tienen al menos 2 bloques de espacio)
        safe_directions = []

        # Verificar dirección derecha
        if grid_x >= 2:
            safe_directions.append(Direction.RIGHT)

        # Verificar dirección izquierda
        if grid_x <= grid_width - 3:
            safe_directions.append(Direction.LEFT)

        # Verificar dirección abajo
        if grid_y >= 2:
            safe_directions.append(Direction.DOWN)

        # Verificar dirección arriba
        if grid_y <= grid_height - 3:
            safe_directions.append(Direction.UP)

        # Si no hay direcciones seguras (muy improbable), usar cualquier dirección
        if not safe_directions:
            safe_directions = list(Direction)

        # Elegir una dirección aleatoria de las seguras
        self.direction = random.choice(safe_directions)

        # Establecer la cabeza
        self.head = Point(x, y)

        # Crear el cuerpo de la serpiente según la dirección elegida
        # con validación adicional para asegurar que esté dentro de los límites
        self.snake = [self.head]

        # Añadir segmentos del cuerpo según la dirección
        if self.direction == Direction.RIGHT:
            self.snake.extend([
                Point(self.head.x - BLOCK_SIZE, self.head.y),
                Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
            ])
        elif self.direction == Direction.LEFT:
            self.snake.extend([
                Point(self.head.x + BLOCK_SIZE, self.head.y),
                Point(self.head.x + (2 * BLOCK_SIZE), self.head.y),
            ])
        elif self.direction == Direction.DOWN:
            self.snake.extend([
                Point(self.head.x, self.head.y - BLOCK_SIZE),
                Point(self.head.x, self.head.y - (2 * BLOCK_SIZE)),
            ])
        elif self.direction == Direction.UP:
            self.snake.extend([
                Point(self.head.x, self.head.y + BLOCK_SIZE),
                Point(self.head.x, self.head.y + (2 * BLOCK_SIZE)),
            ])

        # Verificación final: asegurarse de que todos los segmentos estén dentro de los límites
        # Usar las dimensiones exactas del grid
        grid_width_px = grid_width * BLOCK_SIZE
        grid_height_px = grid_height * BLOCK_SIZE

        for segment in self.snake:
            if (segment.x < 0 or segment.x >= grid_width_px or
                segment.y < 0 or segment.y >= grid_height_px):
                # Si hay algún segmento fuera de los límites, reiniciar el proceso
                print("Advertencia: Serpiente inicializada fuera de los límites. Reintentando...")
                return self.reset()  # Llamada recursiva para reintentar

        # Inicializar variables de juego
        self.score: int = 0
        self.food: Optional[Point] = None
        self.frame_iteration: int = 0

        # Reiniciar variables de seguimiento
        self.reward_history = []
        self.action_history = []
        self.steps = 0
        self.food_distances = []
        self.decision_times = []
        self.game_duration = 0

        # Reiniciar mapa de visitas con tamaño correcto
        grid_width, grid_height = self.grid_size
        self.visit_map = np.zeros((grid_width, grid_height))

        # Inicializar lista de partículas para efectos visuales
        self.particles = []

        # Restaurar configuración visual y dimensiones de la ventana
        self.visual_config = current_visual_config
        self.width = current_width
        self.height = current_height

        # Asegurar que la pantalla mantenga el tamaño correcto
        self.display = pygame.display.set_mode((self.width, self.height))

        # Colocar comida inicial
        self._place_food()

        # Marcar la posición inicial de la serpiente en el mapa de visitas
        for segment in self.snake:
            grid_x, grid_y = self._grid_position(segment)
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                self.visit_map[grid_x, grid_y] = 1

        # Imprimir información de inicialización
        print(f"Juego inicializado: Serpiente en dirección {self.direction.name}, "
              f"Tamaño del grid: {grid_width}x{grid_height}")

    def _place_food(self):
        """Coloca comida en una posición aleatoria que no esté ocupada por la serpiente,
        con validación mejorada de límites y distribución más uniforme."""
        # Crear una matriz de posiciones disponibles para mejor rendimiento
        grid_width, grid_height = self.grid_size
        available_grid = np.ones((grid_width, grid_height), dtype=bool)

        # Marcar posiciones ocupadas por la serpiente
        for segment in self.snake:
            grid_x, grid_y = self._grid_position(segment)
            # Verificar que las coordenadas estén dentro de los límites
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                available_grid[grid_x, grid_y] = False

        # Obtener todas las posiciones disponibles
        available_positions = []
        for x in range(grid_width):
            for y in range(grid_height):
                if available_grid[x, y]:
                    available_positions.append(Point(x * BLOCK_SIZE, y * BLOCK_SIZE))

        # Si no hay posiciones disponibles, el juego está ganado
        if not available_positions:
            print("¡Juego ganado! No hay más espacio disponible.")
            self.food = None
            return

        # Obtener la posición de la cabeza en coordenadas de grid
        head_grid_x, head_grid_y = self._grid_position(self.snake[0])

        # Filtrar posiciones que estén demasiado cerca de la cabeza
        min_distance = 3  # Distancia mínima en unidades de grid
        suitable_positions = []

        for pos in available_positions:
            pos_grid_x, pos_grid_y = self._grid_position(pos)
            manhattan_distance = abs(pos_grid_x - head_grid_x) + abs(pos_grid_y - head_grid_y)

            if manhattan_distance >= min_distance:
                suitable_positions.append(pos)

        # Si no hay posiciones adecuadas, usar cualquier posición disponible
        if not suitable_positions and available_positions:
            suitable_positions = available_positions

        # Elegir una posición con preferencia por áreas menos visitadas
        if suitable_positions:
            # Calcular pesos basados en el mapa de visitas (menos visitas = mayor probabilidad)
            weights = []
            for pos in suitable_positions:
                grid_x, grid_y = self._grid_position(pos)
                # Usar el inverso del número de visitas como peso (más visitas = menor probabilidad)
                visit_count = self.visit_map[grid_x, grid_y] if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height else 0
                # Agregar un pequeño valor para evitar división por cero
                weight = 1.0 / (visit_count + 1.0)
                weights.append(weight)

            # Normalizar pesos
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]

                # Elegir posición basada en los pesos
                # Usar random.choices en lugar de np.random.choice para manejar objetos Point
                self.food = random.choices(suitable_positions, weights=normalized_weights, k=1)[0]
            else:
                # Si hay algún problema con los pesos, elegir aleatoriamente
                self.food = random.choice(suitable_positions)
        else:
            # No debería llegar aquí, pero por seguridad
            print("No se encontraron posiciones adecuadas para la comida.")
            self.food = None
            return

        # Registrar la ubicación de la comida para análisis
        if self.food is not None:
            self.food_locations.append((self.food.x, self.food.y))

            # Imprimir información de depuración
            food_grid_x, food_grid_y = self._grid_position(self.food)

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
            elif event.type == pygame.KEYDOWN:
                # Cambiar modo visual con la tecla 'V'
                if event.key == pygame.K_v:
                    self.toggle_visual_mode()
                # Activar/desactivar pathfinding con la tecla 'P'
                elif event.key == pygame.K_p and "agent" in globals():
                    if hasattr(globals()["agent"], "pathfinding_enabled"):
                        globals()["agent"].set_pathfinding(
                            not globals()["agent"].pathfinding_enabled
                        )
                # Cambiar tamaño de la ventana con la tecla 'S'
                elif event.key == pygame.K_s:
                    self.toggle_window_size()

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

        # Check for game over conditions with detailed collision info
        collision, collision_type = self.get_collision_info()
        timeout = self.frame_iteration > 120 * len(self.snake)

        if collision or timeout:
            game_over = True
            # Higher penalty for early deaths, less penalty for deaths after longer games
            base_penalty = -10.0
            survival_factor = min(
                len(self.snake) / 10.0, 1.0
            )  # Cap at 1, using float division

            # Ajustar la penalización según el tipo de colisión
            if collision_type == 'wall':
                # Mayor penalización por chocar con la pared (error más básico)
                collision_factor = 1.2
            elif collision_type == 'body':
                # Menor penalización por chocar con el cuerpo (más difícil de evitar)
                collision_factor = 1.0
            else:  # timeout
                # Penalización por tiempo agotado (ineficiencia)
                collision_factor = 0.8

            reward = base_penalty * collision_factor * (1 - 0.5 * survival_factor)  # Mantener como float
            self.reward_history.append(reward)
            return reward, game_over, self.score

        # Check for food eaten
        if self.head == self.food:
            self.score += 1
            # Generar partículas de confeti al comer comida
            self.spawn_confetti(self.food)
            # Reward scales with snake length - more reward for growing longer
            base_reward = 1.0
            length_bonus = min(len(self.snake) * 0.5, 10.0)  # Cap at +10 bonus
            reward = base_reward + length_bonus  # Mantener como float
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
            reward = (
                distance_reward
                + survival_reward
                + efficiency_penalty
                + danger_reward
                + future_penalty
            )  # Mantener como float

            # Store reward and remove tail
            self.reward_history.append(reward)
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def toggle_visual_mode(self):
        """Cambia entre los modos de visualización 'animated' y 'simple'."""
        if self.visual_config["visual_mode"] == "animated":
            self.visual_config["visual_mode"] = "simple"
            print("Modo visual cambiado a: Simple")
        else:
            self.visual_config["visual_mode"] = "animated"
            print("Modo visual cambiado a: Animado")

    def toggle_window_size(self):
        """Cambia entre diferentes tamaños de ventana manteniendo el formato de estadio."""
        # Definir tamaños disponibles para la cuadrícula (bloques de ancho, bloques de alto)
        grid_sizes = [(20, 15), (25, 20), (30, 25)]

        # Encontrar el tamaño actual de la cuadrícula en la lista
        current_grid_size = (self.grid_width_blocks, self.grid_height_blocks)
        try:
            current_index = grid_sizes.index(current_grid_size)
            # Cambiar al siguiente tamaño en la lista (circular)
            next_index = (current_index + 1) % len(grid_sizes)
        except ValueError:
            # Si el tamaño actual no está en la lista, usar el primero
            next_index = 0

        # Obtener el nuevo tamaño de cuadrícula
        self.grid_width_blocks, self.grid_height_blocks = grid_sizes[next_index]

        # Recalcular el tamaño total de la ventana
        self.width = self.grid_width_blocks * BLOCK_SIZE + 2 * self.margin_side
        self.height = self.grid_height_blocks * BLOCK_SIZE + self.margin_top + self.margin_bottom

        # Aplicar el nuevo tamaño
        self.display = pygame.display.set_mode((self.width, self.height))

        # Recrear la superficie del mapa de calor con el nuevo tamaño
        self.heatmap_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        print(f"Tamaño de ventana cambiado a: {self.width}x{self.height} (Cuadrícula: {self.grid_width_blocks}x{self.grid_height_blocks})")

        # Reiniciar el juego para adaptarse al nuevo tamaño
        self.reset()

    def spawn_confetti(self, position):
        """Genera partículas de confeti en la posición dada si están habilitadas."""
        # Solo generar partículas si están habilitadas en la configuración
        if not self.visual_config["particle_effects"]:
            return

        num_particles = 20
        for _ in range(num_particles):
            particle = {
                'pos': [self.margin_side + position.x + BLOCK_SIZE // 2, self.margin_top + position.y + BLOCK_SIZE // 2],
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
        """Verifica si hay colisión con los límites o con el cuerpo de la serpiente.
        """
        if point is None:
            point = self.head

        # Obtener dimensiones exactas del grid
        grid_width, grid_height = self.grid_size
        grid_width_px = grid_width * BLOCK_SIZE
        grid_height_px = grid_height * BLOCK_SIZE

        # Validar que el punto esté dentro de los límites válidos
        # Verificar colisión con los bordes usando las dimensiones exactas del grid
        if (
            point.x >= grid_width_px
            or point.x < 0
            or point.y >= grid_height_px
            or point.y < 0
        ):
            return True

        # Verificar colisión con el cuerpo de la serpiente
        # Usar una comparación más precisa para evitar falsos positivos
        for segment in self.snake[1:]:
            if point.x == segment.x and point.y == segment.y:
                return True

        # No hay colisión
        return False

    def get_collision_info(self, point: Optional[Point] = None) -> Tuple[bool, Optional[str]]:
        """Verifica si hay colisión y devuelve información detallada.
        Devuelve una tupla (hay_colisión, tipo_colisión) donde tipo_colisión puede ser:
        'wall' para colisiones con los límites, 'body' para colisiones con el cuerpo,
        o None si no hay colisión.
        """
        if point is None:
            point = self.head

        # Obtener dimensiones exactas del grid
        grid_width, grid_height = self.grid_size
        grid_width_px = grid_width * BLOCK_SIZE
        grid_height_px = grid_height * BLOCK_SIZE

        # Verificar colisión con los bordes usando las dimensiones exactas del grid
        if (
            point.x >= grid_width_px
            or point.x < 0
            or point.y >= grid_height_px
            or point.y < 0
        ):
            return True, 'wall'

        # Verificar colisión con el cuerpo de la serpiente
        for segment in self.snake[1:]:
            if point.x == segment.x and point.y == segment.y:
                return True, 'body'

        # No hay colisión
        return False, None

    def _update_ui(self):
        """Actualiza la interfaz gráfica según el modo de visualización seleccionado."""
        # Llenar toda la pantalla con un color de fondo oscuro
        self.display.fill((20, 20, 30))

        # Dibujar un rectángulo negro para el área de juego (el estadio)
        game_area = pygame.Rect(
            self.margin_side,
            self.margin_top,
            self.grid_width_blocks * BLOCK_SIZE,
            self.grid_height_blocks * BLOCK_SIZE
        )
        pygame.draw.rect(self.display, BLACK, game_area)

        # Dibujar un borde para el estadio
        pygame.draw.rect(self.display, (100, 100, 120), game_area, 3)

        # Dibujar cuadrícula si está habilitada
        if self.visual_config["show_grid"]:
            self._draw_grid()

        # Dibujar mapa de calor si está habilitado
        if self.visual_config["show_heatmap"]:
            self._draw_heatmap()

        # Elegir el método de renderizado según el modo visual
        if self.visual_config["visual_mode"] == "animated":
            self._render_animated()
        else:
            self._render_simple()

        # Renderizar información de juego (común para ambos modos)
        self._render_game_info()

        pygame.display.flip()

    def _draw_grid(self):
        """Dibuja una cuadrícula en el fondo con coordenadas y mejor visibilidad."""
        # Colores mejorados para mejor contraste
        grid_color = (60, 60, 80)  # Gris azulado más visible
        highlight_color = (100, 100, 150)  # Color destacado más brillante
        border_color = (120, 120, 180)  # Color para el borde exterior
        text_color = (180, 180, 220)  # Color para las coordenadas

        # Calcular el número exacto de bloques horizontales y verticales
        grid_width, grid_height = self.grid_size

        # Calcular la posición inicial de la cuadrícula (con márgenes)
        grid_start_x = self.margin_side
        grid_start_y = self.margin_top
        grid_end_x = grid_start_x + grid_width * BLOCK_SIZE
        grid_end_y = grid_start_y + grid_height * BLOCK_SIZE

        # Dibujar solo las líneas necesarias para el grid exacto
        for i in range(grid_width + 1):  # +1 para incluir la línea final
            x = grid_start_x + i * BLOCK_SIZE
            # Línea normal o destacada según posición
            line_color = highlight_color if i % 5 == 0 else grid_color
            line_width = 2 if i % 5 == 0 else 1
            pygame.draw.line(self.display, line_color, (x, grid_start_y), (x, grid_end_y), line_width)

            # Añadir coordenadas X cada 5 bloques
            if i % 5 == 0:
                # Usar una fuente más pequeña para las coordenadas
                try:
                    coord_font = pygame.font.SysFont("arial", 10)
                    coord_text = coord_font.render(str(i), True, text_color)
                    self.display.blit(coord_text, (x + 2, grid_start_y + 2))
                except:
                    pass  # Si hay error con la fuente, omitir coordenadas

        # Líneas horizontales con coordenadas
        for i in range(grid_height + 1):  # +1 para incluir la línea final
            y = grid_start_y + i * BLOCK_SIZE
            # Línea normal o destacada según posición
            line_color = highlight_color if i % 5 == 0 else grid_color
            line_width = 2 if i % 5 == 0 else 1
            pygame.draw.line(self.display, line_color, (grid_start_x, y), (grid_end_x, y), line_width)

            # Añadir coordenadas Y cada 5 bloques
            if i % 5 == 0:
                try:
                    coord_font = pygame.font.SysFont("arial", 10)
                    coord_text = coord_font.render(str(i), True, text_color)
                    self.display.blit(coord_text, (grid_start_x + 2, y + 2))
                except:
                    pass  # Si hay error con la fuente, omitir coordenadas

        # Dibujar indicadores de cuadrante en las esquinas para mejor orientación
        try:
            corner_font = pygame.font.SysFont("arial", 12, bold=True)
            # Esquina superior izquierda
            corner_text = corner_font.render("(0,0)", True, (200, 200, 255))
            self.display.blit(corner_text, (grid_start_x + 5, grid_start_y + 5))

            # Esquina inferior derecha
            max_x = grid_width - 1
            max_y = grid_height - 1
            corner_text = corner_font.render(f"({max_x},{max_y})", True, (200, 200, 255))
            text_width = corner_text.get_width()
            self.display.blit(corner_text, (grid_end_x - text_width - 5, grid_end_y - 20))
        except:
            pass  # Si hay error con la fuente, omitir coordenadas

    def _draw_heatmap(self):
        """Dibuja un mapa de calor de las posiciones visitadas."""
        # Limpiar la superficie del mapa de calor
        self.heatmap_surface.fill((0, 0, 0, 0))

        # Encontrar el valor máximo en el mapa de visitas para normalizar
        max_visits = np.max(self.visit_map) if np.any(self.visit_map) else 1

        # Dibujar rectángulos coloreados según la frecuencia de visitas
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                visits = self.visit_map[x, y]
                if visits > 0:
                    # Normalizar y calcular color (usar un color más sutil)
                    intensity = min(visits / max_visits, 1.0)

                    # Usar la opacidad configurada en config.py
                    alpha = int(HEATMAP_OPACITY * intensity)  # Transparencia configurable

                    # Usar colores diferentes según la intensidad para mejor visualización
                    if intensity < 0.3:
                        # Baja intensidad: azul claro
                        color = (50, 100, 200, alpha)
                    elif intensity < 0.7:
                        # Media intensidad: púrpura
                        color = (100, 50, 200, alpha)
                    else:
                        # Alta intensidad: rojo
                        color = (200, 50, 50, alpha)

                    # Hacer los rectángulos más pequeños para no obstruir la visualización
                    margin = 5
                    rect = pygame.Rect(
                        self.margin_side + x * BLOCK_SIZE + margin,
                        self.margin_top + y * BLOCK_SIZE + margin,
                        BLOCK_SIZE - (margin * 2),
                        BLOCK_SIZE - (margin * 2)
                    )

                    # Dibujar rectángulos con bordes redondeados para un aspecto más suave
                    pygame.draw.rect(self.heatmap_surface, color, rect, border_radius=4)

        # Aplicar la superficie del mapa de calor a la pantalla principal
        self.display.blit(self.heatmap_surface, (0, 0))

    def _render_animated(self):
        """Renderiza el juego con efectos visuales completos."""
        # Actualizar y renderizar partículas (efecto confeti) si están habilitadas
        if self.visual_config["particle_effects"]:
            for particle in self.particles[:]:
                # Actualizar posición y disminuir lifetime
                particle['pos'][0] += particle['vel'][0] * ANIMATION_SPEED
                particle['pos'][1] += particle['vel'][1] * ANIMATION_SPEED
                particle['lifetime'] -= 1 * ANIMATION_SPEED

                # Dibujar partícula
                pygame.draw.circle(
                    self.display,
                    particle['color'],
                    (int(particle['pos'][0]), int(particle['pos'][1])),
                    2
                )

                if particle['lifetime'] <= 0:
                    self.particles.remove(particle)

        # Renderizado mejorado de la serpiente con sombra
        for i, pt in enumerate(self.snake):
            # Definir el rectángulo del segmento (ajustado a los márgenes del estadio)
            snake_rect = pygame.Rect(
                self.margin_side + pt.x,
                self.margin_top + pt.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )

            # Código de sombreado eliminado

            # Dibujar el segmento de la serpiente con degradado de color
            color_factor = 1 - (i / len(self.snake))
            body_color = (
                int(30 + 225 * color_factor),
                int(144 - 39 * color_factor),
                int(255 - 75 * color_factor),
            )

            # Verificar si este segmento está involucrado en una colisión
            is_collision_segment = False
            if i > 0:  # No verificar la cabeza
                # Verificar si la cabeza colisiona con este segmento
                if self.head.x == pt.x and self.head.y == pt.y:
                    is_collision_segment = True

            # Si es un segmento de colisión, dibujar un borde rojo parpadeante
            if is_collision_segment:
                # Efecto parpadeante usando el tiempo
                flash_intensity = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.5
                border_color = (255, int(50 * flash_intensity), int(50 * flash_intensity))

                # Dibujar el segmento con un borde destacado
                pygame.draw.rect(
                    self.display,
                    body_color,
                    snake_rect,
                    border_radius=BLOCK_SIZE // 2
                )
                pygame.draw.rect(
                    self.display,
                    border_color,
                    snake_rect,
                    width=3,
                    border_radius=BLOCK_SIZE // 2
                )
            else:
                # Dibujo normal
                pygame.draw.rect(
                    self.display,
                    body_color,
                    snake_rect,
                    border_radius=BLOCK_SIZE // 2
                )

        # Comida: usar imagen PNG de manzana si está disponible
        if self.food is not None:
            apple_rect = pygame.Rect(
                self.margin_side + self.food.x,
                self.margin_top + self.food.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )

            if self.apple_image:
                # Añadir efecto de pulso a la manzana
                pulse = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.1 + 1.0
                size = int(BLOCK_SIZE * pulse)
                pos_x = self.food.x - (size - BLOCK_SIZE) // 2
                pos_y = self.food.y - (size - BLOCK_SIZE) // 2

                scaled_apple = pygame.transform.scale(self.apple_image, (size, size))
                self.display.blit(scaled_apple, (self.margin_side + pos_x, self.margin_top + pos_y))
            else:
                # Dibujar un círculo rojo con borde
                pygame.draw.circle(
                    self.display,
                    RED,
                    (self.margin_side + self.food.x + BLOCK_SIZE // 2, self.margin_top + self.food.y + BLOCK_SIZE // 2),
                    BLOCK_SIZE // 2
                )
                pygame.draw.circle(
                    self.display,
                    WHITE,
                    (self.margin_side + self.food.x + BLOCK_SIZE // 2, self.margin_top + self.food.y + BLOCK_SIZE // 2),
                    BLOCK_SIZE // 2,
                    2
                )

    def _render_simple(self):
        """Renderiza el juego con gráficos simples para mejor rendimiento."""
        # Dibujar la serpiente (versión simple)
        for pt in self.snake:
            snake_rect = pygame.Rect(
                self.margin_side + pt.x,
                self.margin_top + pt.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
            pygame.draw.rect(self.display, BLUE1, snake_rect)
            pygame.draw.rect(self.display, BLUE2, snake_rect, 1)

        # Dibujar la cabeza con un color diferente
        head_rect = pygame.Rect(
            self.margin_side + self.head.x,
            self.margin_top + self.head.y,
            BLOCK_SIZE,
            BLOCK_SIZE
        )
        pygame.draw.rect(self.display, GREEN, head_rect)
        pygame.draw.rect(self.display, BLACK, head_rect, 1)

        # Dibujar comida (versión simple)
        if self.food is not None:
            food_rect = pygame.Rect(
                self.margin_side + self.food.x,
                self.margin_top + self.food.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
            pygame.draw.rect(self.display, RED, food_rect)
            pygame.draw.rect(self.display, BLACK, food_rect, 1)

    def _render_game_info(self):
        """Renderiza la información del juego en pantalla en formato de marcador."""
        # Crear un marcador en la parte superior
        padding = 10  # Espacio desde el borde

        # Dibujar un fondo para el marcador
        scoreboard_rect = pygame.Rect(
            self.margin_side,
            10,
            self.grid_width_blocks * BLOCK_SIZE,
            self.margin_top - 20
        )
        pygame.draw.rect(self.display, (40, 40, 60), scoreboard_rect)
        pygame.draw.rect(self.display, (100, 100, 140), scoreboard_rect, 2)

        # Crear textos con mejor contraste y sombra para legibilidad
        try:
            info_font = pygame.font.SysFont("arial", 24, bold=True)
        except:
            info_font = self.main_font

        # Crear textos concisos con colores más llamativos
        score_text = info_font.render(f"SCORE: {self.score}", True, (255, 220, 100))
        n_game_text = info_font.render(f"GAME: {self.n_game + 1}", True, (180, 220, 255))
        record_text = info_font.render(f"RECORD: {self.record}", True, (255, 150, 150))

        # Calcular posiciones para alinear horizontalmente los textos en el marcador
        grid_width_px = self.grid_width_blocks * BLOCK_SIZE
        scoreboard_center_y = 10 + (self.margin_top - 20) // 2 - info_font.get_height() // 2

        score_pos = [self.margin_side + padding, scoreboard_center_y]
        game_pos = [self.margin_side + grid_width_px//2 - n_game_text.get_width()//2, scoreboard_center_y]  # Centrado
        record_pos = [self.margin_side + grid_width_px - record_text.get_width() - padding, scoreboard_center_y]  # Derecha

        # Dibujar sombras para mejor legibilidad
        shadow_offset = 2
        shadow_color = (20, 20, 30)

        # Dibujar sombras
        shadow_text = score_text.copy()
        self.display.blit(shadow_text, [score_pos[0] + shadow_offset, score_pos[1] + shadow_offset])
        shadow_text = n_game_text.copy()
        self.display.blit(shadow_text, [game_pos[0] + shadow_offset, game_pos[1] + shadow_offset])
        shadow_text = record_text.copy()
        self.display.blit(shadow_text, [record_pos[0] + shadow_offset, record_pos[1] + shadow_offset])

        # Dibujar textos
        self.display.blit(score_text, score_pos)
        self.display.blit(n_game_text, game_pos)
        self.display.blit(record_text, record_pos)

        # Visualizar colisiones con los bordes si la cabeza está cerca del borde
        self._visualize_border_collisions()

        # Añadir indicador de modo visual y pathfinding en una línea pequeña en la parte inferior
        if "agent" in globals() and hasattr(globals()["agent"], "pathfinding_enabled"):
            # Crear texto de estado en la parte inferior
            status_font = pygame.font.SysFont("arial", 14) if pygame.font.get_init() else self.small_font

            # Crear texto de estado
            pathfinding_status = "ON" if globals()["agent"].pathfinding_enabled else "OFF"
            mode_name = self.visual_config['visual_mode'].capitalize()

            status_text = status_font.render(
                f"Mode: {mode_name} | Pathfinding: {pathfinding_status} | 'V': Change Mode | 'P': Toggle Pathfinding | 'S': Change Size",
                True,
                (200, 200, 200)
            )

            # Posicionar en la parte inferior usando el tamaño exacto del grid
            grid_width, grid_height = self.grid_size
            grid_width_px = grid_width * BLOCK_SIZE
            grid_height_px = grid_height * BLOCK_SIZE

            status_pos = [self.margin_side + grid_width_px//2 - status_text.get_width()//2,
                          self.margin_top + grid_height_px - status_text.get_height() - 5]

            # Dibujar con sombra para legibilidad
            self.display.blit(status_text.copy(), [status_pos[0] + 1, status_pos[1] + 1])
            self.display.blit(status_text, status_pos)

        # El código para mostrar el estado de pathfinding y modo visual
        # ahora está integrado en la sección anterior

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
        return (self.grid_width_blocks, self.grid_height_blocks)

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

    def _visualize_border_collisions(self):
        """Visualiza posibles colisiones con los bordes cuando la cabeza está cerca."""
        # Distancia de advertencia en píxeles
        warning_distance = BLOCK_SIZE * 2

        # Obtener dimensiones exactas del grid
        grid_width, grid_height = self.grid_size
        grid_width_px = grid_width * BLOCK_SIZE
        grid_height_px = grid_height * BLOCK_SIZE

        # Verificar distancia a cada borde (considerando la posición real en la cuadrícula)
        near_left = self.head.x < warning_distance
        near_right = self.head.x > grid_width_px - warning_distance - BLOCK_SIZE
        near_top = self.head.y < warning_distance
        near_bottom = self.head.y > grid_height_px - warning_distance - BLOCK_SIZE

        # Si la cabeza está cerca de algún borde, mostrar advertencia visual
        if near_left or near_right or near_top or near_bottom:
            # Calcular intensidad del efecto basado en la proximidad
            # Usar un efecto parpadeante
            flash_intensity = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.5
            warning_color = (255, int(100 + 155 * flash_intensity), 0, int(100 + 155 * flash_intensity))

            # Crear una superficie con transparencia para el efecto de advertencia
            warning_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

            # Dibujar rectángulos de advertencia en los bordes cercanos
            border_width = 5  # Ancho del borde de advertencia

            # Calcular las coordenadas del área de juego con los márgenes
            game_left = self.margin_side
            game_top = self.margin_top
            game_right = game_left + grid_width_px
            game_bottom = game_top + grid_height_px

            if near_left:
                pygame.draw.rect(warning_surface, warning_color, (game_left, game_top, border_width, grid_height_px))
            if near_right:
                pygame.draw.rect(warning_surface, warning_color, (game_right - border_width, game_top, border_width, grid_height_px))
            if near_top:
                pygame.draw.rect(warning_surface, warning_color, (game_left, game_top, grid_width_px, border_width))
            if near_bottom:
                pygame.draw.rect(warning_surface, warning_color, (game_left, game_bottom - border_width, grid_width_px, border_width))

            # Aplicar la superficie de advertencia a la pantalla principal
            self.display.blit(warning_surface, (0, 0))

    def toggle_visual_mode(self):
        """Cambia entre los modos de visualización animado y simple."""
        if self.visual_config["visual_mode"] == "animated":
            self.visual_config["visual_mode"] = "simple"
        else:
            self.visual_config["visual_mode"] = "animated"

        # Mostrar mensaje de cambio de modo
        print(f"Modo visual cambiado a: {self.visual_config['visual_mode']}")

    def update_visual_config(self, config):
        """Actualiza la configuración visual con nuevos valores."""
        if config:
            self.visual_config.update(config)
