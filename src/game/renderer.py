"""
Sistema de renderizado para el juego Snake.
Este módulo proporciona las funciones para dibujar todos los elementos visuales del juego.

Componentes principales:
- Renderizado de la serpiente y la comida
- Visualización de la cuadrícula
- Mapa de calor para posiciones visitadas
- Efectos visuales avanzados
"""

import pygame
import numpy as np
from utils.config import (
    BLOCK_SIZE, BLUE1, BLUE2, RED, WHITE, BLACK, GREEN, YELLOW, GRAY, PURPLE,
    HEATMAP_OPACITY, ANIMATION_SPEED
)

class Renderer:
    """Clase responsable de renderizar todos los elementos visuales del juego."""
    
    def __init__(self, game):
        """
        Inicializa el sistema de renderizado.
        
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

        # Cargar imagen de manzana
        try:
            self.apple_image = pygame.image.load("assets/apple.png").convert_alpha()
            self.apple_image = pygame.transform.scale(self.apple_image, (BLOCK_SIZE, BLOCK_SIZE))
        except Exception as e:
            print("Error loading apple image from assets/apple.png, using fallback drawing.")
            self.apple_image = None
            
        # Crear superficie para el mapa de calor
        self.heatmap_surface = pygame.Surface((game.width, game.height), pygame.SRCALPHA)
    
    def update_ui(self):
        """Actualiza la interfaz gráfica según el modo de visualización seleccionado."""
        # Llenar toda la pantalla con un color de fondo oscuro
        self.game.display.fill((20, 20, 30))

        # Dibujar un rectángulo negro para el área de juego (el estadio)
        game_area = pygame.Rect(
            self.game.margin_side,
            self.game.margin_top,
            self.game.grid_width_blocks * BLOCK_SIZE,
            self.game.grid_height_blocks * BLOCK_SIZE
        )
        pygame.draw.rect(self.game.display, BLACK, game_area)

        # Dibujar un borde para el estadio
        pygame.draw.rect(self.game.display, (100, 100, 120), game_area, 3)

        # Dibujar cuadrícula si está habilitada
        if self.game.visual_config["show_grid"]:
            self._draw_grid()

        # Dibujar mapa de calor si está habilitado
        if self.game.visual_config["show_heatmap"]:
            self._draw_heatmap()

        # Elegir el método de renderizado según el modo visual
        if self.game.visual_config["visual_mode"] == "animated":
            self._render_animated()
        else:
            self._render_simple()

        # Renderizar información de juego (común para ambos modos)
        self.game.ui.render_game_info()

        pygame.display.flip()
    
    def _draw_grid(self):
        """Dibuja una cuadrícula en el fondo con coordenadas y mejor visibilidad."""
        # Colores mejorados para mejor contraste
        grid_color = (60, 60, 80)  # Gris azulado más visible
        highlight_color = (100, 100, 150)  # Color destacado más brillante
        border_color = (120, 120, 180)  # Color para el borde exterior
        text_color = (180, 180, 220)  # Color para las coordenadas

        # Calcular el número exacto de bloques horizontales y verticales
        grid_width, grid_height = self.game.grid_size

        # Calcular la posición inicial de la cuadrícula (con márgenes)
        grid_start_x = self.game.margin_side
        grid_start_y = self.game.margin_top
        grid_end_x = grid_start_x + grid_width * BLOCK_SIZE
        grid_end_y = grid_start_y + grid_height * BLOCK_SIZE

        # Dibujar solo las líneas necesarias para el grid exacto
        for i in range(grid_width + 1):  # +1 para incluir la línea final
            x = grid_start_x + i * BLOCK_SIZE
            # Línea normal o destacada según posición
            line_color = highlight_color if i % 5 == 0 else grid_color
            line_width = 2 if i % 5 == 0 else 1
            pygame.draw.line(self.game.display, line_color, (x, grid_start_y), (x, grid_end_y), line_width)

            # Añadir coordenadas X cada 5 bloques
            if i % 5 == 0:
                # Usar una fuente más pequeña para las coordenadas
                try:
                    coord_font = pygame.font.SysFont("arial", 10)
                    coord_text = coord_font.render(str(i), True, text_color)
                    self.game.display.blit(coord_text, (x + 2, grid_start_y + 2))
                except:
                    pass  # Si hay error con la fuente, omitir coordenadas

        # Líneas horizontales con coordenadas
        for i in range(grid_height + 1):  # +1 para incluir la línea final
            y = grid_start_y + i * BLOCK_SIZE
            # Línea normal o destacada según posición
            line_color = highlight_color if i % 5 == 0 else grid_color
            line_width = 2 if i % 5 == 0 else 1
            pygame.draw.line(self.game.display, line_color, (grid_start_x, y), (grid_end_x, y), line_width)

            # Añadir coordenadas Y cada 5 bloques
            if i % 5 == 0:
                try:
                    coord_font = pygame.font.SysFont("arial", 10)
                    coord_text = coord_font.render(str(i), True, text_color)
                    self.game.display.blit(coord_text, (grid_start_x + 2, y + 2))
                except:
                    pass  # Si hay error con la fuente, omitir coordenadas

        # Dibujar indicadores de cuadrante en las esquinas para mejor orientación
        try:
            corner_font = pygame.font.SysFont("arial", 12, bold=True)
            # Esquina superior izquierda
            corner_text = corner_font.render("(0,0)", True, (200, 200, 255))
            self.game.display.blit(corner_text, (grid_start_x + 5, grid_start_y + 5))

            # Esquina inferior derecha
            max_x = grid_width - 1
            max_y = grid_height - 1
            corner_text = corner_font.render(f"({max_x},{max_y})", True, (200, 200, 255))
            text_width = corner_text.get_width()
            self.game.display.blit(corner_text, (grid_end_x - text_width - 5, grid_end_y - 20))
        except:
            pass  # Si hay error con la fuente, omitir coordenadas
    
    def _draw_heatmap(self):
        """Dibuja un mapa de calor de las posiciones visitadas."""
        # Limpiar la superficie del mapa de calor
        self.heatmap_surface.fill((0, 0, 0, 0))

        # Encontrar el valor máximo en el mapa de visitas para normalizar
        max_visits = np.max(self.game.visit_map) if np.any(self.game.visit_map) else 1

        # Dibujar rectángulos coloreados según la frecuencia de visitas
        for x in range(self.game.grid_size[0]):
            for y in range(self.game.grid_size[1]):
                visits = self.game.visit_map[x, y]
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
                        self.game.margin_side + x * BLOCK_SIZE + margin,
                        self.game.margin_top + y * BLOCK_SIZE + margin,
                        BLOCK_SIZE - (margin * 2),
                        BLOCK_SIZE - (margin * 2)
                    )

                    # Dibujar rectángulos con bordes redondeados para un aspecto más suave
                    pygame.draw.rect(self.heatmap_surface, color, rect, border_radius=4)

        # Aplicar la superficie del mapa de calor a la pantalla principal
        self.game.display.blit(self.heatmap_surface, (0, 0))
    
    def _render_animated(self):
        """Renderiza el juego con efectos visuales completos."""
        # Actualizar y renderizar partículas (efecto confeti) si están habilitadas
        if self.game.visual_config["particle_effects"]:
            for particle in self.game.particles[:]:
                # Actualizar posición y disminuir lifetime
                particle['pos'][0] += particle['vel'][0] * ANIMATION_SPEED
                particle['pos'][1] += particle['vel'][1] * ANIMATION_SPEED
                particle['lifetime'] -= 1 * ANIMATION_SPEED

                # Dibujar partícula
                pygame.draw.circle(
                    self.game.display,
                    particle['color'],
                    (int(particle['pos'][0]), int(particle['pos'][1])),
                    2
                )

                if particle['lifetime'] <= 0:
                    self.game.particles.remove(particle)

        # Renderizado mejorado de la serpiente con sombra
        for i, pt in enumerate(self.game.snake):
            # Definir el rectángulo del segmento (ajustado a los márgenes del estadio)
            snake_rect = pygame.Rect(
                self.game.margin_side + pt.x,
                self.game.margin_top + pt.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )

            # Dibujar sombra si está habilitada
            if self.game.visual_config["shadow_effects"]:
                shadow_offset = 3
                shadow_color = (50, 50, 50)
                shadow_rect = snake_rect.copy()
                shadow_rect.x += shadow_offset
                shadow_rect.y += shadow_offset
                pygame.draw.rect(
                    self.game.display,
                    shadow_color,
                    shadow_rect,
                    border_radius=BLOCK_SIZE // 2
                )

            # Dibujar el segmento de la serpiente con degradado de color
            color_factor = 1 - (i / len(self.game.snake))
            body_color = (
                int(30 + 225 * color_factor),
                int(144 - 39 * color_factor),
                int(255 - 75 * color_factor),
            )

            # Verificar si este segmento está involucrado en una colisión
            is_collision_segment = False
            if i > 0:  # No verificar la cabeza
                # Verificar si la cabeza colisiona con este segmento
                if self.game.head.x == pt.x and self.game.head.y == pt.y:
                    is_collision_segment = True

            # Si es un segmento de colisión, dibujar un borde rojo parpadeante
            if is_collision_segment:
                # Efecto parpadeante usando el tiempo
                flash_intensity = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.5
                border_color = (255, int(50 * flash_intensity), int(50 * flash_intensity))

                # Dibujar el segmento con un borde destacado
                pygame.draw.rect(
                    self.game.display,
                    body_color,
                    snake_rect,
                    border_radius=BLOCK_SIZE // 2
                )
                pygame.draw.rect(
                    self.game.display,
                    border_color,
                    snake_rect,
                    width=3,
                    border_radius=BLOCK_SIZE // 2
                )
            else:
                # Dibujo normal
                pygame.draw.rect(
                    self.game.display,
                    body_color,
                    snake_rect,
                    border_radius=BLOCK_SIZE // 2
                )

        # Comida: usar imagen PNG de manzana si está disponible
        if self.game.food is not None:
            apple_rect = pygame.Rect(
                self.game.margin_side + self.game.food.x,
                self.game.margin_top + self.game.food.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )

            if self.apple_image:
                # Añadir efecto de pulso a la manzana
                pulse = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.1 + 1.0
                size = int(BLOCK_SIZE * pulse)
                pos_x = self.game.food.x - (size - BLOCK_SIZE) // 2
                pos_y = self.game.food.y - (size - BLOCK_SIZE) // 2

                scaled_apple = pygame.transform.scale(self.apple_image, (size, size))
                self.game.display.blit(scaled_apple, (self.game.margin_side + pos_x, self.game.margin_top + pos_y))
            else:
                # Dibujar un círculo rojo con borde
                pygame.draw.circle(
                    self.game.display,
                    RED,
                    (self.game.margin_side + self.game.food.x + BLOCK_SIZE // 2, self.game.margin_top + self.game.food.y + BLOCK_SIZE // 2),
                    BLOCK_SIZE // 2
                )
                pygame.draw.circle(
                    self.game.display,
                    WHITE,
                    (self.game.margin_side + self.game.food.x + BLOCK_SIZE // 2, self.game.margin_top + self.game.food.y + BLOCK_SIZE // 2),
                    BLOCK_SIZE // 2,
                    2
                )
    
    def _render_simple(self):
        """Renderiza el juego con gráficos simples para mejor rendimiento."""
        # Dibujar la serpiente (versión simple)
        for pt in self.game.snake:
            snake_rect = pygame.Rect(
                self.game.margin_side + pt.x,
                self.game.margin_top + pt.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
            pygame.draw.rect(self.game.display, BLUE1, snake_rect)
            pygame.draw.rect(self.game.display, BLUE2, snake_rect, 1)

        # Dibujar la cabeza con un color diferente
        head_rect = pygame.Rect(
            self.game.margin_side + self.game.head.x,
            self.game.margin_top + self.game.head.y,
            BLOCK_SIZE,
            BLOCK_SIZE
        )
        pygame.draw.rect(self.game.display, GREEN, head_rect)
        pygame.draw.rect(self.game.display, BLACK, head_rect, 1)

        # Dibujar comida (versión simple)
        if self.game.food is not None:
            food_rect = pygame.Rect(
                self.game.margin_side + self.game.food.x,
                self.game.margin_top + self.game.food.y,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
            pygame.draw.rect(self.game.display, RED, food_rect)
            pygame.draw.rect(self.game.display, BLACK, food_rect, 1)
