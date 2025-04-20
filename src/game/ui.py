"""
Módulo para la interfaz de usuario del juego Snake.
Este módulo proporciona elementos visuales informativos como marcadores y estadísticas.

Componentes principales:
- Visualización de puntuación y estadísticas
- Marcadores y paneles informativos
- Indicadores visuales de estado
"""

import pygame
import numpy as np
from utils.config import BLOCK_SIZE

class UI:
    """Clase para manejar los elementos de interfaz de usuario."""

    def __init__(self, game):
        """
        Inicializa la interfaz de usuario.

        Args:
            game: Referencia a la instancia principal del juego
        """
        self.game = game

        # Cargar fuentes
        try:
            self.main_font = pygame.font.Font("assets/arial.ttf", 25)
            self.small_font = pygame.font.Font("assets/arial.ttf", 15)
        except FileNotFoundError:
            print("No se encontró 'assets/arial.ttf'. Usando fuente predeterminada.")
            self.main_font = pygame.font.SysFont("arial", 25)
            self.small_font = pygame.font.SysFont("arial", 15)

    def render_game_info(self):
        """Renderiza la información del juego en pantalla en formato de marcador."""
        # Crear un marcador en la parte superior
        padding = 10  # Espacio desde el borde

        # Dibujar un fondo para el marcador
        scoreboard_rect = pygame.Rect(
            self.game.margin_side,
            10,
            self.game.grid_width_blocks * BLOCK_SIZE,
            self.game.margin_top - 20
        )
        pygame.draw.rect(self.game.display, (40, 40, 60), scoreboard_rect)
        pygame.draw.rect(self.game.display, (100, 100, 140), scoreboard_rect, 2)

        # Crear textos con mejor contraste y sombra para legibilidad
        try:
            info_font = pygame.font.SysFont("arial", 24, bold=True)
        except:
            info_font = self.main_font

        # Crear textos concisos con colores más llamativos
        score_text = info_font.render(f"SCORE: {self.game.score}", True, (255, 220, 100))
        n_game_text = info_font.render(f"GAME: {self.game.n_game + 1}", True, (180, 220, 255))
        record_text = info_font.render(f"RECORD: {self.game.record}", True, (255, 150, 150))

        # Calcular posiciones para alinear horizontalmente los textos en el marcador
        grid_width_px = self.game.grid_width_blocks * BLOCK_SIZE
        scoreboard_center_y = 10 + (self.game.margin_top - 20) // 2 - info_font.get_height() // 2

        score_pos = [self.game.margin_side + padding, scoreboard_center_y]
        game_pos = [self.game.margin_side + grid_width_px//2 - n_game_text.get_width()//2, scoreboard_center_y]  # Centrado
        record_pos = [self.game.margin_side + grid_width_px - record_text.get_width() - padding, scoreboard_center_y]  # Derecha

        # Dibujar sombras para mejor legibilidad
        shadow_offset = 2
        shadow_color = (20, 20, 30)

        # Dibujar sombras
        shadow_text = score_text.copy()
        self.game.display.blit(shadow_text, [score_pos[0] + shadow_offset, score_pos[1] + shadow_offset])
        shadow_text = n_game_text.copy()
        self.game.display.blit(shadow_text, [game_pos[0] + shadow_offset, game_pos[1] + shadow_offset])
        shadow_text = record_text.copy()
        self.game.display.blit(shadow_text, [record_pos[0] + shadow_offset, record_pos[1] + shadow_offset])

        # Dibujar textos
        self.game.display.blit(score_text, score_pos)
        self.game.display.blit(n_game_text, game_pos)
        self.game.display.blit(record_text, record_pos)

        # Visualizar colisiones con los bordes si la cabeza está cerca del borde
        self._visualize_border_collisions()

        # Añadir indicador de modo visual y controles en una línea pequeña en la parte inferior
        # Crear texto de estado en la parte inferior
        status_font = pygame.font.SysFont("arial", 14) if pygame.font.get_init() else self.small_font

        # Crear texto de estado
        mode_name = self.game.visual_config['visual_mode'].capitalize()

        # Intentar obtener el estado de pathfinding si existe
        pathfinding_status = "N/A"
        try:
            import builtins
            agent = getattr(builtins, 'agent', None)
            if agent and hasattr(agent, "pathfinding_enabled"):
                pathfinding_status = "ON" if agent.pathfinding_enabled else "OFF"
        except Exception:
            pass

        status_text = status_font.render(
            f"Mode: {mode_name} | Pathfinding: {pathfinding_status} | 'V': Change Mode | 'P': Toggle Pathfinding | 'S': Change Size",
            True,
            (200, 200, 200)
        )

        # Posicionar en la parte inferior usando el tamaño exacto del grid
        grid_width, grid_height = self.game.grid_size
        grid_width_px = grid_width * BLOCK_SIZE
        grid_height_px = grid_height * BLOCK_SIZE

        status_pos = [self.game.margin_side + grid_width_px//2 - status_text.get_width()//2,
                      self.game.margin_top + grid_height_px - status_text.get_height() - 5]

        # Dibujar con sombra para legibilidad
        self.game.display.blit(status_text.copy(), [status_pos[0] + 1, status_pos[1] + 1])
        self.game.display.blit(status_text, status_pos)

    def _visualize_border_collisions(self):
        """Visualiza posibles colisiones con los bordes cuando la cabeza está cerca."""
        # Distancia de advertencia en píxeles
        warning_distance = BLOCK_SIZE * 2

        # Obtener dimensiones exactas del grid
        grid_width, grid_height = self.game.grid_size
        grid_width_px = grid_width * BLOCK_SIZE
        grid_height_px = grid_height * BLOCK_SIZE

        # Verificar distancia a cada borde (considerando la posición real en la cuadrícula)
        near_left = self.game.head.x < warning_distance
        near_right = self.game.head.x > grid_width_px - warning_distance - BLOCK_SIZE
        near_top = self.game.head.y < warning_distance
        near_bottom = self.game.head.y > grid_height_px - warning_distance - BLOCK_SIZE

        # Si la cabeza está cerca de algún borde, mostrar advertencia visual
        if near_left or near_right or near_top or near_bottom:
            # Calcular intensidad del efecto basado en la proximidad
            # Usar un efecto parpadeante
            flash_intensity = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.5
            warning_color = (255, int(100 + 155 * flash_intensity), 0, int(100 + 155 * flash_intensity))

            # Crear una superficie con transparencia para el efecto de advertencia
            warning_surface = pygame.Surface((self.game.width, self.game.height), pygame.SRCALPHA)

            # Dibujar rectángulos de advertencia en los bordes cercanos
            border_width = 5  # Ancho del borde de advertencia

            # Calcular las coordenadas del área de juego con los márgenes
            game_left = self.game.margin_side
            game_top = self.game.margin_top
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
            self.game.display.blit(warning_surface, (0, 0))
