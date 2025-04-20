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
from typing import Optional, Tuple, List, Dict, Any
from utils.config import (
    MAX_MEMORY,
    LR,
    GAMMA,
    TEMPERATURE,
    EXPLORATION_PHASE,
    MIN_TEMPERATURE,
    DECAY_RATE,
    EXPLORATION_FREQUENCY,
    EXPLORATION_TEMP,
    EXPLORATION_DURATION,
    MAX_EPOCHS,
    BLOCK_SIZE,
    BATCH_SIZE,
    TAU,
    MEMORY_PRUNE_FACTOR,
    MEMORY_MONITOR_FREQ,
    MEMORY_PRUNE_THRESHOLD,
    ANIMATION_SPEED,
    STADIUM_MARGIN_TOP,
    STADIUM_MARGIN_SIDE,
    STADIUM_MARGIN_BOTTOM,
    WHITE, RED, BLUE1, BLUE2, BLACK, GREEN, YELLOW, GRAY, PURPLE
)
from utils.advanced_pathfinding import AdvancedPathfinding
from src.stats_manager import StatsManager
import time
from utils.helper import event_system

pygame.init()

try:
    font = pygame.font.Font("assets/arial.ttf", 25)
except FileNotFoundError:
    print("No se encontró 'assets/arial.ttf'. Usando fuente predeterminada.")
    font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SnakeGameAI:
    def __init__(
        self, width: int = 1300, height: int = 750, n_game: int = 0, record: int = 0,
        agent=None,
        event_system_instance=None
    ) -> None:
        # Definir el ancho del panel lateral de estadísticas
        self.stats_panel_width = 450

        # Calcular el tamaño de la cuadrícula de juego (sin márgenes)
        self.grid_width_blocks = (width - 2 * STADIUM_MARGIN_SIDE - self.stats_panel_width) // BLOCK_SIZE
        self.grid_height_blocks = (height - STADIUM_MARGIN_TOP - STADIUM_MARGIN_BOTTOM) // BLOCK_SIZE

        # Recalcular el tamaño total de la ventana para que se ajuste perfectamente
        self.width: int = self.grid_width_blocks * BLOCK_SIZE + 2 * STADIUM_MARGIN_SIDE + self.stats_panel_width
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
        self.visual_config = {
            "show_stats_panel": True,
            "selected_stats": ["basic", "training", "efficiency", "actions", "model"]
        }

        # Cargar fuentes
        try:
            self.main_font = pygame.font.Font("assets/arial.ttf", 25)
            self.small_font = pygame.font.Font("assets/arial.ttf", 15)
            self.stats_title_font = pygame.font.Font("assets/arial.ttf", 18)
            self.stats_font = pygame.font.Font("assets/arial.ttf", 14)
        except FileNotFoundError:
            print("No se encontró 'assets/arial.ttf'. Usando fuente predeterminada.")
            self.main_font = pygame.font.SysFont("arial", 25)
            self.small_font = pygame.font.SysFont("arial", 15)
            self.stats_title_font = pygame.font.SysFont("arial", 18, bold=True)
            self.stats_font = pygame.font.SysFont("arial", 14)

        # Cargar imagen de manzana
        try:
            self.apple_image = pygame.image.load("assets/apple.png").convert_alpha()
            self.apple_image = pygame.transform.scale(self.apple_image, (BLOCK_SIZE, BLOCK_SIZE))
        except Exception as e:
            print("Error loading apple image from assets/apple.png, using fallback drawing.")
            self.apple_image = None

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

        # --- NUEVO: Para calcular pasos por comida ---
        self.steps_per_food = []
        self.steps_since_last_food = 0

        # Instancia de StatsManager
        if event_system_instance is None:
            event_system_instance = event_system
        self.event_system = event_system_instance
        self.stats_manager = StatsManager(self.event_system, self, agent)

        # Inicializar variables de juego
        self.score: int = 0
        self.food: Optional[Point] = None
        self.frame_iteration: int = 0
        # Inicializar la serpiente antes de cualquier uso
        self.snake = [Point(self.grid_width_blocks // 2 * BLOCK_SIZE, self.grid_height_blocks // 2 * BLOCK_SIZE)]
        self.head = self.snake[0]
        self.direction = Direction.RIGHT

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
        self.visual_config = {
            "show_stats_panel": True,
            "selected_stats": ["basic", "training", "efficiency", "actions", "model"]
        }
        self.width = width
        self.height = height

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

        # Actualizar el resumen del juego al reiniciar
        try:
            from utils.helper import update_game_summary
            from src.shared_data import get_model_params

            # Obtener los parámetros del modelo
            model_params = get_model_params()

            # Inicializar la categoría "model" con los parámetros obtenidos
            if model_params:
                model_stats = {
                    "Temperatura": model_params.get("temperature", 0.99),
                    "Learning rate": model_params.get("learning_rate", 0.001),
                    "Pathfinding": "Activado" if model_params.get("pathfinding_enabled", True) else "Desactivado",
                    "Modo de explotación": model_params.get("mode", "Pathfinding habilitado")
                }
                self.stats_manager.update()

            # Actualizar el resumen del juego
            agent = globals()["agent"] if "agent" in globals() else None
            if agent:
                # Actualizar estadísticas de entrenamiento con el último récord
                if "training" not in self.stats_manager.get_stats():
                    self.stats_manager.update()
                # Usar nuestro método dedicado para obtener el valor más actualizado
                last_record = self.stats_manager.get_last_record_game()
                self.stats_manager.update()
                # Comentado para no saturar la consola
                # print(f"[DEBUG] reset: Último récord = {last_record}")

                # Forzar una actualización del resumen del juego
                summary = update_game_summary(game=self, agent=agent, force_update=True)
                if summary:
                    # Notificar a través del sistema de eventos
                    self.event_system.notify("game_summary_updated", summary)
        except Exception:
            # Ignorar errores silenciosamente para no saturar la consola
            pass

        # Lógica reactiva para el panel de estadísticas
        self.stats_panel_needs_update = True
        self.event_system.register_listener("stats_updated", self.on_stats_updated)

    def on_stats_updated(self, data):
        """Marca que el panel de estadísticas debe actualizarse tras un cambio."""
        self.stats_panel_needs_update = True

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

        # --- NUEVO: Reiniciar pasos por comida ---
        self.steps_per_food = []
        self.steps_since_last_food = 0

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

        # Notificar que se debe actualizar el panel de estadísticas tras reinicio
        self.event_system.notify("stats_update_needed", {
            "score": self.score,
            "record": self.record
        })

        # Actualizar el resumen del juego al reiniciar
        try:
            from utils.helper import update_game_summary
            from src.shared_data import get_model_params

            # Obtener los parámetros del modelo
            model_params = get_model_params()

            # Inicializar la categoría "model" con los parámetros obtenidos
            if model_params:
                model_stats = {
                    "Temperatura": model_params.get("temperature", 0.99),
                    "Learning rate": model_params.get("learning_rate", 0.001),
                    "Pathfinding": "Activado" if model_params.get("pathfinding_enabled", True) else "Desactivado",
                    "Modo de explotación": model_params.get("mode", "Pathfinding habilitado")
                }
                self.stats_manager.update()

            # Actualizar el resumen del juego
            agent = globals()["agent"] if "agent" in globals() else None
            if agent:
                # Actualizar estadísticas de entrenamiento con el último récord
                if "training" not in self.stats_manager.get_stats():
                    self.stats_manager.update()
                # Usar nuestro método dedicado para obtener el valor más actualizado
                last_record = self.stats_manager.get_last_record_game()
                self.stats_manager.update()
                # Comentado para no saturar la consola
                # print(f"[DEBUG] reset: Último récord = {last_record}")

                # Forzar una actualización del resumen del juego
                summary = update_game_summary(game=self, agent=agent, force_update=True)
                if summary:
                    # Notificar a través del sistema de eventos
                    self.event_system.notify("game_summary_updated", summary)
        except Exception:
            # Ignorar errores silenciosamente para no saturar la consola
            pass

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
        self.steps_since_last_food += 1
        self.n_game = n_game
        self.record = record
        self.frame_iteration += 1

        # IMPORTANTE: Actualizar el valor del último récord en cada paso
        # Esto garantiza que siempre tengamos el valor más actualizado
        if "training" in self.stats_manager.get_stats():
            last_record = self.stats_manager.get_last_record_game()
            self.stats_manager.update()
            # Comentado para no saturar la consola
            # if self.steps % 100 == 0:
            #     print(f"[DEBUG] play_step: Último récord = {last_record}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                # Cambiar tamaño de la ventana con la tecla 'S'
                if event.key == pygame.K_s:
                    self.toggle_window_size()
                # Alternar panel de estadísticas con la tecla 'T'
                elif event.key == pygame.K_t:
                    self.toggle_stats_panel()

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
            # --- NUEVO: Registrar pasos por comida ---
            self.steps_per_food.append(self.steps_since_last_food)
            self.steps_since_last_food = 0
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

        # Detectar si la puntuación o el récord han cambiado y notificar
        score_changed = False
        record_changed = False
        prev_score = getattr(self, '_prev_score', None)
        prev_record = getattr(self, '_prev_record', None)
        if prev_score is None or self.score != prev_score:
            score_changed = True
            self._prev_score = self.score
        if prev_record is None or self.record != prev_record:
            record_changed = True
            self._prev_record = self.record
        if score_changed or record_changed:
            self.event_system.notify("stats_update_needed", {
                "score": self.score,
                "record": self.record
            })

        # Mantener la funcionalidad de cálculo de normas de pesos para otros componentes
        # pero sin mostrarlas en el panel
        try:
            from utils.helper import print_weight_norms
            agent = globals()["agent"] if "agent" in globals() else None
            if agent and hasattr(agent, "model"):
                # Calcular las normas de pesos para otros componentes que puedan necesitarlas
                # pero no mostrarlas en el panel
                print_weight_norms(agent)

                # Actualizar las estadísticas (sin normas de pesos)
                self.stats_manager.update()
        except Exception:
            # Ignorar errores silenciosamente para no saturar la consola
            pass

        self._update_ui()
        self.clock.tick(60)
        return reward, game_over, self.score

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
        # Solo actualizar estadísticas si el panel realmente necesita refresco
        if self.stats_panel_needs_update or self.stats_manager.is_dirty():
            self.stats_manager.update()
            self.stats_manager.clean()
            self.stats_panel_needs_update = False
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

        # Dibujar mapa de calor si está habilitado
        self._draw_heatmap()

        # Renderizado animado
        self._render_animated()

        # Renderizar información de juego (común para ambos modos)
        self._render_game_info()

        # Renderizar panel de estadísticas solo si hay cambios
        if self.visual_config.get("show_stats_panel", True):
            self._render_stats_panel()

        pygame.display.flip()

    def _render_stats_panel(self):
        """Renderiza el panel de estadísticas con layout robusto: sin empalmes, con bloques y alineación dinámica según cantidad de elementos."""
        panel_x = self.margin_side + self.grid_width_blocks * BLOCK_SIZE + 20
        panel_y = self.margin_top
        panel_width = self.stats_panel_width - 30
        panel_height = self.grid_height_blocks * BLOCK_SIZE
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        if not hasattr(self, '_stats_panel_static_cache'):
            self._stats_panel_static_cache = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            self._stats_panel_static_cache_dirty = True
        if not hasattr(self, '_stats_panel_value_cache'):
            self._stats_panel_value_cache = {}
            self._stats_panel_value_anim_time = {}
        ANIMATION_DURATION = 0.35
        now = time.time()

        # Colores y estilos mejorados para visualización agradable
        bg_color = (18, 22, 34)          # Fondo: azul muy oscuro
        border_color = (70, 140, 200)    # Borde: azul medio suave
        title_color = (180, 220, 255)    # Títulos: celeste claro
        line_color = (50, 70, 100)       # Líneas divisorias: gris azulado
        label_color = (200, 210, 230)    # Etiquetas: gris claro
        value_color = (110, 210, 255)    # Valores: azul/celeste
        highlight_color = (255, 210, 90) # Récords: dorado suave
        warning_color = (255, 170, 80)   # Advertencias: naranja suave
        padding = 18
        section_spacing = 16
        value_spacing = 22
        block_spacing = 22
        section_line_width = 1
        font = self.stats_font
        title_font = self.stats_title_font

        # Obtener dinámicamente la cantidad de valores por categoría
        stats = self.stats_manager.get_stats()
        categories = ["basic", "training", "efficiency", "model"]
        category_titles = [self._get_category_title(c) for c in categories]
        values_per_category = [len(stats.get(cat, {})) for cat in categories]

        # Calcular offsets para cada bloque para evitar empalmes
        y_offsets = []
        y = padding
        for idx, n_vals in enumerate(values_per_category):
            y_offsets.append(y)
            y += title_font.get_height() + section_spacing
            y += n_vals * value_spacing
            # Línea divisoria entre bloques
            if idx < len(categories) - 1:
                y += block_spacing

        if self._stats_panel_static_cache_dirty:
            self._stats_panel_static_cache.fill((0, 0, 0, 0))
            pygame.draw.rect(self._stats_panel_static_cache, bg_color, (0, 0, panel_width, panel_height), border_radius=12)
            pygame.draw.rect(self._stats_panel_static_cache, border_color, (0, 0, panel_width, panel_height), 2, border_radius=12)
            for idx, category in enumerate(categories):
                y_offset = y_offsets[idx]
                cat_title = title_font.render(category_titles[idx], True, title_color)
                self._stats_panel_static_cache.blit(cat_title, (padding, y_offset))
                y_offset += title_font.get_height() + section_spacing - 4
                # Línea divisoria tras título
                pygame.draw.line(self._stats_panel_static_cache, line_color,
                                 (padding, y_offset),
                                 (panel_width - padding, y_offset), section_line_width)
                # Línea divisoria entre bloques
                if idx < len(categories) - 1:
                    next_block_y = y_offsets[idx+1] - int(block_spacing/2)
                    pygame.draw.line(self._stats_panel_static_cache, line_color,
                                     (padding, next_block_y),
                                     (panel_width - padding, next_block_y), section_line_width)
            self._stats_panel_static_cache_dirty = False

        self.display.blit(self._stats_panel_static_cache, (panel_x, panel_y))

        # Render dinámico de valores, alineado según layout calculado
        for idx, category in enumerate(categories):
            y_offset = panel_y + y_offsets[idx] + title_font.get_height() + section_spacing
            if category in stats:
                for k, v in stats[category].items():
                    if k in ["Récord", "Último récord (juego)", "Modo de explotación"]:
                        color = highlight_color
                    elif k in ["Pathfinding"]:
                        color = (120, 255, 170) if v == "Activado" else warning_color
                    elif k in ["Pérdida"] and isinstance(v, float) and v > 3.0:
                        color = warning_color
                    elif isinstance(v, float):
                        color = value_color
                    else:
                        color = label_color
                    cache_key = f"{category}:{k}"
                    prev_val = self._stats_panel_value_cache.get(cache_key, v)
                    last_anim_time = self._stats_panel_value_anim_time.get(cache_key, now)
                    if isinstance(v, float) or isinstance(v, int):
                        if prev_val != v:
                            self._stats_panel_value_anim_time[cache_key] = now
                            start_val = prev_val
                        else:
                            start_val = prev_val
                        elapsed = min((now - last_anim_time) / ANIMATION_DURATION, 1.0)
                        if elapsed < 1.0:
                            display_val = start_val + (v - start_val) * elapsed
                        else:
                            display_val = v
                        self._stats_panel_value_cache[cache_key] = display_val
                        if abs(display_val - v) > 1e-3:
                            self.stats_panel_needs_update = True
                        value_str = f"{display_val:.4f}" if abs(display_val) < 100 else f"{display_val:.2f}"
                    else:
                        value_str = str(v)
                        self._stats_panel_value_cache[cache_key] = v
                        self._stats_panel_value_anim_time[cache_key] = now
                    text = font.render(f"{k}: ", True, label_color)
                    val = font.render(value_str, True, color)
                    self.display.blit(text, (panel_x + padding, y_offset))
                    self.display.blit(val, (panel_x + padding + text.get_width() + 6, y_offset))
                    y_offset += value_spacing

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
                    alpha = int(255 * intensity)  # Transparencia configurable

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

            # Dibujar sombra si está habilitada
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
            mode_name = "Animado"

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

    def _get_category_title(self, category):
        """Devuelve el título formateado para una categoría de estadísticas."""
        titles = {
            "basic": "Estadísticas Básicas",
            "training": "Entrenamiento",
            "efficiency": "Eficiencia",
            "actions": "Distribución de Acciones",
            "model": "Parámetros del Modelo"
        }
        return titles.get(category, category.capitalize())

    def _get_last_record_game(self):
        """Obtiene el valor de last_record_game directamente del agente.
        Este método garantiza que siempre obtendremos el valor más actualizado.

        Returns:
            int: El número del juego donde se obtuvo el último récord.
        """
        # Importar el sistema de registro de errores
        from utils.error_logger import stats_panel_logger, log_error, log_warning, log_info

        try:
            # Intentar obtener el agente global
            agent = globals().get("agent", None)
            if agent is None:
                log_warning(stats_panel_logger, "LastRecord", "No se pudo obtener el agente global")
            elif not hasattr(agent, "last_record_game"):
                log_warning(stats_panel_logger, "LastRecord", "El agente no tiene el atributo 'last_record_game'")
            else:
                # Agente válido con el atributo necesario
                last_record = agent.last_record_game
                log_info(stats_panel_logger, "LastRecord", f"Obtenido último récord del agente: {last_record}")
                return last_record

            # Si no se puede obtener del agente, intentar obtenerlo de shared_data
            from utils.helper import latest_game_summary
            if "Último récord obtenido en partida" in latest_game_summary:
                last_record = latest_game_summary["Último récord obtenido en partida"]
                log_info(stats_panel_logger, "LastRecord", f"Obtenido último récord de latest_game_summary: {last_record}")
                return last_record
            else:
                log_warning(stats_panel_logger, "LastRecord", "No se encontró el último récord en latest_game_summary")

        except Exception as e:
            # Registrar el error con detalles
            log_error(
                stats_panel_logger,
                "LastRecord",
                "Error al obtener último récord",
                exception=e,
                context={
                    "agent_exists": "agent" in globals(),
                    "latest_game_summary_keys": str(list(latest_game_summary.keys())) if 'latest_game_summary' in locals() else "No disponible"
                }
            )

        # Si no se pudo obtener de ninguna fuente, devolver 0
        return 0

    def toggle_stats_panel(self):
        """Alterna la visibilidad del panel de estadísticas."""
        # Simplemente alternar entre mostrar y ocultar el panel
        self.visual_config["show_stats_panel"] = not self.visual_config.get("show_stats_panel", True)

        status = "activado" if self.visual_config["show_stats_panel"] else "desactivado"
        print(f"Panel de estadísticas {status}")

    def update_visual_config(self, config):
        """Actualiza la configuración visual con nuevos valores."""
        if config:
            self.visual_config.update(config)

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
