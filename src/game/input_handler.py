"""
Módulo para el manejo de entrada del usuario en el juego Snake.
Este módulo procesa los eventos de teclado y ratón para controlar el juego.

Componentes principales:
- Procesamiento de eventos de pygame
- Manejo de teclas para cambiar modos visuales
- Control de tamaño de ventana
"""

import pygame
import sys
import builtins

class InputHandler:
    """Clase para manejar la entrada del usuario en el juego."""

    def __init__(self, game):
        """
        Inicializa el manejador de entrada.

        Args:
            game: Referencia a la instancia principal del juego
        """
        self.game = game

    def handle_events(self):
        """Procesa todos los eventos de entrada del usuario."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                self._handle_key_press(event.key)

    def _handle_key_press(self, key):
        """
        Maneja las pulsaciones de teclas específicas.

        Args:
            key: Código de la tecla presionada
        """
        # Cambiar modo visual con la tecla 'V'
        if key == pygame.K_v:
            self.game.toggle_visual_mode()

        # Activar/desactivar pathfinding con la tecla 'P'
        elif key == pygame.K_p:
            # Intentar acceder a la variable global 'agent'
            try:
                agent = getattr(builtins, 'agent', None)
                if agent and hasattr(agent, "pathfinding_enabled"):
                    agent.set_pathfinding(not agent.pathfinding_enabled)
            except Exception as e:
                print(f"Error al acceder a 'agent': {e}")

        # Cambiar tamaño de la ventana con la tecla 'S'
        elif key == pygame.K_s:
            self.game.toggle_window_size()
