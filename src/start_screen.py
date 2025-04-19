"""
Módulo para la pantalla de inicio del juego Snake DQN.
Permite al usuario seleccionar entre diferentes modos de visualización
antes de iniciar el entrenamiento.
"""

import pygame
import sys
from utils.config import (
    WHITE, BLACK, BLUE1, BLUE2, GREEN, RED, GRAY,
    VISUAL_MODE, SHOW_GRID, SHOW_HEATMAP, PARTICLE_EFFECTS, SHADOW_EFFECTS
)
from utils.config_manager import save_visual_config, load_visual_config

class Button:
    """Clase para crear botones interactivos en la pantalla."""

    def __init__(self, x, y, width, height, text, color, hover_color, text_color=WHITE, font_size=24):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font_size = font_size
        self.is_hovered = False

        # Crear fuente
        try:
            self.font = pygame.font.Font("assets/arial.ttf", font_size)
        except FileNotFoundError:
            self.font = pygame.font.SysFont("arial", font_size)

    def draw(self, surface):
        """Dibuja el botón en la superficie dada."""
        # Dibujar el fondo del botón
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=10)  # Borde

        # Dibujar el texto
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        """Actualiza el estado del botón según la posición del mouse."""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos, mouse_click):
        """Comprueba si el botón ha sido clicado."""
        return self.rect.collidepoint(mouse_pos) and mouse_click


class Checkbox:
    """Clase para crear casillas de verificación."""

    def __init__(self, x, y, size, text, is_checked=False, text_color=WHITE, font_size=20):
        self.rect = pygame.Rect(x, y, size, size)
        self.text = text
        self.is_checked = is_checked
        self.text_color = text_color
        self.size = size
        self.hover = False

        # Crear fuente
        try:
            self.font = pygame.font.Font("assets/arial.ttf", font_size)
        except FileNotFoundError:
            self.font = pygame.font.SysFont("arial", font_size)

        # Calcular rectángulo del área clickeable (incluye texto)
        text_width = self.font.size(text)[0]
        self.click_rect = pygame.Rect(x, y, size + text_width + 15, size)

    def draw(self, surface):
        """Dibuja la casilla de verificación en la superficie dada."""
        # Dibujar el fondo si está en hover
        if self.hover:
            bg_rect = self.click_rect.inflate(10, 6)
            pygame.draw.rect(surface, (50, 50, 50), bg_rect, border_radius=5)

        # Dibujar el cuadro
        pygame.draw.rect(surface, WHITE, self.rect, 2)

        # Dibujar la marca si está seleccionada
        if self.is_checked:
            inner_rect = pygame.Rect(
                self.rect.x + 4,
                self.rect.y + 4,
                self.rect.width - 8,
                self.rect.height - 8
            )
            pygame.draw.rect(surface, GREEN, inner_rect)

        # Dibujar el texto
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos, mouse_click):
        """Actualiza el estado de la casilla según la interacción del usuario."""
        self.hover = self.click_rect.collidepoint(mouse_pos)

        if self.click_rect.collidepoint(mouse_pos) and mouse_click:
            self.is_checked = not self.is_checked
            return True
        return False


class StartScreen:
    """Clase para la pantalla de inicio del juego."""

    def __init__(self, width=800, height=600):
        # Inicializar pygame si no está inicializado
        if not pygame.get_init():
            pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake DQN - Configuración")

        # Cargar configuración guardada o usar valores por defecto
        saved_config = load_visual_config()
        self.visual_mode = saved_config.get("visual_mode", VISUAL_MODE)
        self.show_grid = saved_config.get("show_grid", SHOW_GRID)
        self.show_heatmap = saved_config.get("show_heatmap", SHOW_HEATMAP)
        self.particle_effects = saved_config.get("particle_effects", PARTICLE_EFFECTS)
        self.shadow_effects = saved_config.get("shadow_effects", SHADOW_EFFECTS)

        # Crear botones
        button_width, button_height = 200, 50
        center_x = width // 2

        self.animated_button = Button(
            center_x - button_width - 20,
            height // 2 - 30,
            button_width,
            button_height,
            "Modo Animado",
            BLUE1,
            BLUE2
        )

        self.simple_button = Button(
            center_x + 20,
            height // 2 - 30,
            button_width,
            button_height,
            "Modo Simple",
            BLUE1,
            BLUE2
        )

        self.start_button = Button(
            center_x - button_width // 2,
            height - 100,
            button_width,
            button_height,
            "Iniciar",
            GREEN,
            (100, 255, 100)
        )

        # Crear checkboxes
        checkbox_y_start = height // 2 + 70
        checkbox_size = 24
        checkbox_spacing = 45
        left_column_x = width // 2 - 200
        right_column_x = width // 2 + 20

        self.grid_checkbox = Checkbox(
            left_column_x,
            checkbox_y_start,
            checkbox_size,
            "Mostrar Cuadrícula",
            self.show_grid
        )

        self.heatmap_checkbox = Checkbox(
            left_column_x,
            checkbox_y_start + checkbox_spacing,
            checkbox_size,
            "Mostrar Mapa de Calor",
            self.show_heatmap
        )

        self.particles_checkbox = Checkbox(
            right_column_x,
            checkbox_y_start,
            checkbox_size,
            "Efectos de Partículas",
            self.particle_effects
        )

        self.shadows_checkbox = Checkbox(
            right_column_x,
            checkbox_y_start + checkbox_spacing,
            checkbox_size,
            "Efectos de Sombra",
            self.shadow_effects
        )

        # Crear fuentes
        try:
            self.title_font = pygame.font.Font("assets/arial.ttf", 48)
            self.subtitle_font = pygame.font.Font("assets/arial.ttf", 24)
        except FileNotFoundError:
            self.title_font = pygame.font.SysFont("arial", 48)
            self.subtitle_font = pygame.font.SysFont("arial", 24)

    def run(self):
        """Ejecuta la pantalla de inicio y devuelve la configuración seleccionada."""
        clock = pygame.time.Clock()
        running = True

        while running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Botón izquierdo
                        mouse_click = True

            # Actualizar botones
            self.animated_button.update(mouse_pos)
            self.simple_button.update(mouse_pos)
            self.start_button.update(mouse_pos)

            # Comprobar clics en botones
            if self.animated_button.is_clicked(mouse_pos, mouse_click):
                self.visual_mode = "animated"

            if self.simple_button.is_clicked(mouse_pos, mouse_click):
                self.visual_mode = "simple"

            if self.start_button.is_clicked(mouse_pos, mouse_click):
                # Guardar configuración y salir
                return {
                    "visual_mode": self.visual_mode,
                    "show_grid": self.grid_checkbox.is_checked,
                    "show_heatmap": self.heatmap_checkbox.is_checked,
                    "particle_effects": self.particles_checkbox.is_checked,
                    "shadow_effects": self.shadows_checkbox.is_checked
                }

            # Actualizar checkboxes
            self.grid_checkbox.update(mouse_pos, mouse_click)
            self.heatmap_checkbox.update(mouse_pos, mouse_click)
            self.particles_checkbox.update(mouse_pos, mouse_click)
            self.shadows_checkbox.update(mouse_pos, mouse_click)

            # Dibujar pantalla
            self.draw()

            pygame.display.flip()
            clock.tick(60)

    def draw(self):
        """Dibuja todos los elementos de la pantalla de inicio."""
        # Fondo
        self.screen.fill(BLACK)

        # Título con efecto de sombra
        shadow_offset = 2
        title_shadow = self.title_font.render("Snake DQN", True, (50, 50, 50))
        title_shadow_rect = title_shadow.get_rect(center=(self.width // 2 + shadow_offset, 80 + shadow_offset))
        self.screen.blit(title_shadow, title_shadow_rect)

        title_surf = self.title_font.render("Snake DQN", True, WHITE)
        title_rect = title_surf.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_surf, title_rect)

        # Subtítulo
        subtitle_surf = self.subtitle_font.render("Selecciona el modo de visualización", True, WHITE)
        subtitle_rect = subtitle_surf.get_rect(center=(self.width // 2, 140))
        self.screen.blit(subtitle_surf, subtitle_rect)

        # Panel para los botones de modo
        mode_panel = pygame.Rect(self.width // 2 - 230, self.height // 2 - 50, 460, 80)
        pygame.draw.rect(self.screen, (30, 30, 30), mode_panel, border_radius=15)
        pygame.draw.rect(self.screen, (70, 70, 70), mode_panel, 2, border_radius=15)

        # Resaltar el botón del modo seleccionado
        if self.visual_mode == "animated":
            pygame.draw.rect(self.screen, GREEN, self.animated_button.rect.inflate(10, 10), 3, border_radius=12)
        else:
            pygame.draw.rect(self.screen, GREEN, self.simple_button.rect.inflate(10, 10), 3, border_radius=12)

        # Dibujar botones
        self.animated_button.draw(self.screen)
        self.simple_button.draw(self.screen)
        self.start_button.draw(self.screen)

        # Panel para las opciones
        options_panel = pygame.Rect(self.width // 2 - 230, self.height // 2 + 40, 460, 130)
        pygame.draw.rect(self.screen, (30, 30, 30), options_panel, border_radius=15)
        pygame.draw.rect(self.screen, (70, 70, 70), options_panel, 2, border_radius=15)

        # Dibujar sección de opciones
        options_text = self.subtitle_font.render("Opciones de Visualización", True, WHITE)
        options_rect = options_text.get_rect(center=(self.width // 2, self.height // 2 + 55))
        self.screen.blit(options_text, options_rect)

        # Dibujar checkboxes
        self.grid_checkbox.draw(self.screen)
        self.heatmap_checkbox.draw(self.screen)
        self.particles_checkbox.draw(self.screen)
        self.shadows_checkbox.draw(self.screen)


def get_user_config():
    """Función para obtener la configuración del usuario desde la pantalla de inicio.
    Guarda la configuración seleccionada para futuras sesiones.
    """
    start_screen = StartScreen()
    config = start_screen.run()

    # Guardar la configuración seleccionada
    save_visual_config(config)

    return config


if __name__ == "__main__":
    # Prueba de la pantalla de inicio
    pygame.init()
    config = get_user_config()
    print("Configuración seleccionada:", config)
    pygame.quit()
