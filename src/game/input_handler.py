import pygame
import sys


class InputHandler:
    def __init__(self, game):
        self.game = game

    def handle_input(self):
        """
        Maneja los eventos de entrada de Pygame.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                # Cambiar tamaño de la ventana con la tecla 'S'
                if event.key == pygame.K_s:
                    self.game.toggle_window_size()
                # Alternar panel de estadísticas con la tecla 'T'
                elif event.key == pygame.K_t:
                    self.game.toggle_stats_panel()
                # Tecla 'P' para alternar pathfinding (si existe el agente global)
                elif event.key == pygame.K_p:
                    agent = globals().get("agent", None)
                    if agent and hasattr(agent, "toggle_pathfinding"):
                        agent.toggle_pathfinding()
