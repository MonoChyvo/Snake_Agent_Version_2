WHITE = (255, 255, 255)
RED = (220, 20, 60)
BLUE1 = (30, 144, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

MAX_EPOCHS = 2_500
BLOCK_SIZE = 20
SPEED = 60

MAX_MEMORY = 150_000
LR = 0.002
GAMMA = 0.96
BATCH_SIZE = 512
TAU = 0.002

TEMPERATURE = 0.61
MIN_TEMPERATURE = 0.05
PREV_LOSS = 0.0
DECAY_RATE = 0.9992

EXPLORATION_PHASE = False
EXPLORATION_FREQUENCY = 100
EXPLORATION_TEMP = 0.89
EXPLORATION_DURATION = 10

# --------- NUEVOS PARÁMETROS ---------

# Sistema de alertas: umbrales para métricas clave
ALERT_THRESHOLDS = {
    'loss': {'high': 1.0, 'critical': 2.0},  # Valores altos indican problemas
    'avg_reward': {'low': -0.5, 'critical': -1.0},  # Valores bajos indican problemas
    'efficiency_ratio': {'low': 0.6},  # Bajo ratio de eficiencia indica movimiento ineficiente
    'steps_per_food': {'high': 100},  # Muchos pasos por comida indica ineficiencia
    'weight_norm_ratio': {'high': 3.0, 'critical': 5.0},  # Ratios muy diferentes indican desbalance
}