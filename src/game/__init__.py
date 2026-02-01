"""
Paquete principal del juego Snake para Aprendizaje Q Profundo.
Este paquete proporciona la mecánica del juego y visualización para la IA de Snake.

Módulos:
- core: Lógica principal del juego
- renderer: Sistema de renderizado
- input_handler: Manejo de entrada del usuario
- ui: Elementos de interfaz de usuario
- effects: Efectos visuales y partículas
"""

from src.game.core import SnakeGameAI, Direction, Point

# Exportar clases y funciones para mantener la API pública
__all__ = ['SnakeGameAI', 'Direction', 'Point']
