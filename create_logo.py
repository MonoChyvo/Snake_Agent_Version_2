"""
Script para crear un logo simple para el proyecto Snake DQN.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
import os

# Crear figura con fondo transparente
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_alpha(0.0)
ax.set_aspect('equal')

# Configurar ejes
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axis('off')

# Colores
snake_color = '#30A0FF'  # Azul
food_color = '#DC143C'   # Rojo
bg_color = '#202030'     # Azul oscuro

# Dibujar fondo circular
bg = Circle((0, 0), 4.5, color=bg_color, alpha=0.9, zorder=1)
ax.add_patch(bg)

# Dibujar serpiente (cuerpo)
snake_segments = [
    (-2, -2), (-1, -2), (0, -2), (0, -1), (0, 0), 
    (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)
]

# Dibujar segmentos con degradado de color
for i, (x, y) in enumerate(snake_segments):
    color_factor = 0.5 + 0.5 * (1 - i / len(snake_segments))
    segment_color = (
        30/255 + 225/255 * color_factor,
        144/255 - 39/255 * color_factor,
        255/255 - 75/255 * color_factor,
    )
    segment = Rectangle(
        (x - 0.4, y - 0.4), 
        0.8, 0.8, 
        color=segment_color,
        alpha=0.9,
        zorder=2,
        ec='black',
        lw=1,
        capstyle='round',
        joinstyle='round'
    )
    ax.add_patch(segment)

# Dibujar comida (manzana)
food = Circle((3, -2), 0.4, color=food_color, alpha=0.9, zorder=2)
ax.add_patch(food)

# Añadir texto "DQN"
plt.text(
    0, -3.5, 
    "SNAKE DQN", 
    fontsize=24, 
    fontweight='bold', 
    color='white', 
    ha='center', 
    va='center',
    zorder=3
)

# Guardar imagen
os.makedirs('assets', exist_ok=True)
plt.savefig('assets/snake_dqn_logo.png', dpi=200, bbox_inches='tight', transparent=True)
print("Logo creado en assets/snake_dqn_logo.png")
