"""
Módulo para efectos visuales en el juego Snake.
Este módulo proporciona efectos visuales como partículas y sombras.

Componentes principales:
- Sistema de partículas para efectos de confeti
- Efectos de sombra para la serpiente
- Animaciones y efectos visuales
"""

import random
from utils.config import BLOCK_SIZE

class Effects:
    """Clase para manejar los efectos visuales del juego."""
    
    def __init__(self, game):
        """
        Inicializa el sistema de efectos visuales.
        
        Args:
            game: Referencia a la instancia principal del juego
        """
        self.game = game
    
    def spawn_confetti(self, position):
        """
        Genera partículas de confeti en la posición dada si están habilitadas.
        
        Args:
            position: Posición donde generar las partículas
        """
        # Solo generar partículas si están habilitadas en la configuración
        if not self.game.visual_config["particle_effects"]:
            return

        num_particles = 20
        for _ in range(num_particles):
            particle = {
                'pos': [self.game.margin_side + position.x + BLOCK_SIZE // 2, self.game.margin_top + position.y + BLOCK_SIZE // 2],
                'vel': [random.uniform(-2, 2), random.uniform(-2, 2)],
                'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                'lifetime': random.uniform(20, 40)
            }
            self.game.particles.append(particle)
