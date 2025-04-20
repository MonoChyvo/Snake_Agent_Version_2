"""
Módulo para gestionar la configuración del juego Snake DQN.
Permite guardar y cargar la configuración visual entre sesiones.
Implementa validación de datos para garantizar la integridad de la configuración.
"""

import os
import json
import logging
from utils.config import (
    VISUAL_MODE,
    SHOW_GRID,
    SHOW_HEATMAP,
    PARTICLE_EFFECTS,
    SHADOW_EFFECTS
)
from utils.validation import validate_config, safe_json_load

CONFIG_FILE = "config.json"

def save_visual_config(config):
    """
    Guarda la configuración visual en un archivo JSON con validación.

    Args:
        config: Diccionario con la configuración visual

    Returns:
        bool: True si la configuración se guardó correctamente

    Raises:
        ValueError: Si la configuración es inválida
    """
    try:
        # Validar la configuración antes de guardarla
        validate_config(config)

        # Guardar la configuración
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuración guardada en {CONFIG_FILE}")
        return True
    except ValueError as e:
        # Propagar errores de validación
        logging.error(f"Error de validación al guardar la configuración: {e}")
        raise
    except Exception as e:
        logging.error(f"Error al guardar la configuración: {e}")
        return False

def load_visual_config():
    """
    Carga la configuración visual desde un archivo JSON con validación.
    Si el archivo no existe o es inválido, devuelve la configuración por defecto.

    Returns:
        dict: Configuración visual validada
    """
    default_config = {
        "visual_mode": VISUAL_MODE,
        "show_grid": SHOW_GRID,
        "show_heatmap": SHOW_HEATMAP,
        "particle_effects": PARTICLE_EFFECTS,
        "shadow_effects": SHADOW_EFFECTS
    }

    if not os.path.exists(CONFIG_FILE):
        logging.info(f"Archivo de configuración {CONFIG_FILE} no encontrado. Usando valores por defecto.")
        return default_config

    try:
        # Cargar y validar la configuración
        config = safe_json_load(CONFIG_FILE)
        validate_config(config)
        logging.info(f"Configuración cargada y validada desde {CONFIG_FILE}")
        return config
    except FileNotFoundError:
        logging.warning(f"Archivo de configuración {CONFIG_FILE} no encontrado. Usando valores por defecto.")
        return default_config
    except json.JSONDecodeError as e:
        logging.error(f"Formato JSON inválido en {CONFIG_FILE}: {e}. Usando valores por defecto.")
        return default_config
    except ValueError as e:
        logging.error(f"Configuración inválida en {CONFIG_FILE}: {e}. Usando valores por defecto.")
        return default_config
    except Exception as e:
        logging.error(f"Error inesperado al cargar la configuración: {e}. Usando valores por defecto.")
        return default_config
