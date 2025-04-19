"""
Módulo para gestionar la configuración del juego Snake DQN.
Permite guardar y cargar la configuración visual entre sesiones.
"""

import os
import json
from utils.config import (
    VISUAL_MODE,
    SHOW_GRID,
    SHOW_HEATMAP,
    PARTICLE_EFFECTS,
    SHADOW_EFFECTS
)

CONFIG_FILE = "config.json"

def save_visual_config(config):
    """
    Guarda la configuración visual en un archivo JSON.
    
    Args:
        config: Diccionario con la configuración visual
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuración guardada en {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"Error al guardar la configuración: {e}")
        return False

def load_visual_config():
    """
    Carga la configuración visual desde un archivo JSON.
    Si el archivo no existe, devuelve la configuración por defecto.
    
    Returns:
        dict: Configuración visual
    """
    default_config = {
        "visual_mode": VISUAL_MODE,
        "show_grid": SHOW_GRID,
        "show_heatmap": SHOW_HEATMAP,
        "particle_effects": PARTICLE_EFFECTS,
        "shadow_effects": SHADOW_EFFECTS
    }
    
    if not os.path.exists(CONFIG_FILE):
        print(f"Archivo de configuración {CONFIG_FILE} no encontrado. Usando valores por defecto.")
        return default_config
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        print(f"Configuración cargada desde {CONFIG_FILE}")
        return config
    except Exception as e:
        print(f"Error al cargar la configuración: {e}")
        return default_config
