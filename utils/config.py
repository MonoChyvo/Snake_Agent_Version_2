"""
Archivo de configuración para la implementación del Snake DQN (Red Q Profunda).

Este módulo contiene todos los hiperparámetros y constantes utilizados en el proyecto:
- Colores de visualización y configuración visual
- Parámetros de entrenamiento (épocas, tasa de aprendizaje, tamaño de lote)
- Configuración de memoria y búfer de repetición
- Parámetros de temperatura y exploración
- Umbrales de alerta para monitorear el rendimiento del entrenamiento

Versión: 1.0.0
"""

# Colores básicos
WHITE = (255, 255, 255)
RED = (220, 20, 60)
BLUE1 = (30, 144, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (50, 205, 50)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (128, 128, 128)

# Parámetros de juego
MAX_EPOCHS = 4_000
BLOCK_SIZE = 30  # Aumentado de 25 a 30 para un grid aún más grande
SPEED = 40  # Reducido de 60 a 40 para una velocidad más lenta

# Parámetros de estadio
STADIUM_MARGIN_TOP = 60  # Margen superior para el marcador
STADIUM_MARGIN_SIDE = 50  # Margen lateral para centrar la cuadrícula
STADIUM_MARGIN_BOTTOM = 30  # Margen inferior

# Configuración de visualización
# Solo se conserva lo relevante para la versión animada
ANIMATION_SPEED = 1.0  # Velocidad de animación (1.0 = normal)

# Eliminados: VISUAL_MODE, SHOW_GRID, SHOW_HEATMAP, PARTICLE_EFFECTS, SHADOW_EFFECTS, HEATMAP_OPACITY

# Parámetros de gestión de memoria
MAX_MEMORY = 300_000
MEMORY_MONITOR_FREQ = 1000  # Verificar uso de memoria cada X juegos
MEMORY_PRUNE_THRESHOLD = 0.9  # Podar cuando el búfer esté al 90% de capacidad
MEMORY_PRUNE_FACTOR = 0.7  # Mantener el 70% de las experiencias después de podar

# Hiperparámetros de entrenamiento
LR = 0.001  # Tasa de aprendizaje
GAMMA = 0.999  # Factor de descuento
BATCH_SIZE = 2048  # Tamaño del lote
TAU = 0.005  # Tasa de actualización de la red objetivo

# Parámetros de temperatura y exploración
TEMPERATURE = 0.99  # Temperatura inicial
MIN_TEMPERATURE = 0.15  # Temperatura mínima
PREV_LOSS = 0.0  # Pérdida previa
DECAY_RATE = 0.9992  # Tasa de decaimiento

# Configuración de fase de exploración
EXPLORATION_PHASE = False  # Activar/desactivar fase de exploración
EXPLORATION_FREQUENCY = 75  # Frecuencia de fases de exploración
EXPLORATION_TEMP = 0.95  # Temperatura durante exploración
EXPLORATION_DURATION = 15  # Duración de la fase de exploración

# Sistema de alertas: umbrales para métricas clave
ALERT_THRESHOLDS = {
    "loss": {"high": 1.5, "critical": 2.5},  # Mayor tolerancia para escenarios complejos
    "avg_reward": {"low": -0.8, "critical": -1.5},  # Ajustado para episodios más largos
    "efficiency_ratio": {"low": 0.5},  # Relajado para serpiente más larga
    "steps_per_food": {"high": 150},  # Aumentado para escenarios de serpiente más larga
    "weight_norm_ratio": {
        "high": 3.5,
        "critical": 5.5,
    },  # Ajustado para patrones más complejos
}
