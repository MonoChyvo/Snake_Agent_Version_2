"""
Archivo de configuración para la implementación del Snake DQN (Red Q Profunda).
Contiene todos los hiperparámetros y constantes utilizados en el proyecto:
- Colores de visualización y configuración visual
- Parámetros de entrenamiento (épocas, tasa de aprendizaje, tamaño de lote)
- Configuración de memoria y búfer de repetición
- Parámetros de temperatura y exploración
- Umbrales de alerta para monitorear el rendimiento del entrenamiento
"""

WHITE = (255, 255, 255)
RED = (220, 20, 60)
BLUE1 = (30, 144, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

MAX_EPOCHS = 4_000
BLOCK_SIZE = 20
SPEED = 60

# Memory management parameters
MAX_MEMORY = 300_000
MEMORY_MONITOR_FREQ = 1000  # Check memory usage every X games
MEMORY_PRUNE_THRESHOLD = 0.9  # Prune when buffer is 90% full
MEMORY_PRUNE_FACTOR = 0.7  # Keep 70% of experiences after pruning

LR = 0.0008
GAMMA = 0.995
BATCH_SIZE = 2048
TAU = 0.003

TEMPERATURE = 0.71
MIN_TEMPERATURE = 0.15
PREV_LOSS = 0.0
DECAY_RATE = 0.9992

EXPLORATION_PHASE = False
EXPLORATION_FREQUENCY = 75
EXPLORATION_TEMP = 0.95
EXPLORATION_DURATION = 15


# Sistema de alertas: umbrales para métricas clave
ALERT_THRESHOLDS = {
    "loss": {"high": 1.5, "critical": 2.5},  # Increased tolerance for complex scenarios
    "avg_reward": {"low": -0.8, "critical": -1.5},  # Adjusted for longer episodes
    "efficiency_ratio": {"low": 0.5},  # Relaxed for longer snake
    "steps_per_food": {"high": 150},  # Increased for longer snake scenarios
    "weight_norm_ratio": {
        "high": 3.5,
        "critical": 5.5,
    },  # Adjusted for more complex patterns
}
