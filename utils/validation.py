"""
Módulo de validación para el proyecto Snake DQN.

Este módulo proporciona funciones para validar diferentes tipos de datos
y entradas externas, asegurando la integridad y seguridad del sistema.

Incluye validación para:
- Archivos de configuración JSON
- Modelos guardados
- Archivos CSV de resultados
- Recursos (imágenes, fuentes)
"""

import os
import json
import csv
import logging
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("security.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("security")

def validate_config(config_data: Dict[str, Any]) -> bool:
    """
    Valida que los datos de configuración tengan la estructura y tipos correctos.
    
    Args:
        config_data (dict): Datos de configuración cargados desde JSON
        
    Returns:
        bool: True si la configuración es válida
        
    Raises:
        ValueError: Si la configuración contiene valores inválidos
    """
    required_keys = ["visual_mode", "show_grid", "show_heatmap", 
                     "particle_effects", "shadow_effects"]
    
    # Verificar que config_data sea un diccionario
    if not isinstance(config_data, dict):
        error_msg = f"Configuración inválida: se esperaba un diccionario, se recibió {type(config_data)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que todas las claves requeridas existan
    for key in required_keys:
        if key not in config_data:
            error_msg = f"Configuración inválida: falta la clave '{key}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Verificar tipos de datos
    if not isinstance(config_data["visual_mode"], str):
        error_msg = f"visual_mode debe ser una cadena de texto, se recibió {type(config_data['visual_mode'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    for key in ["show_grid", "show_heatmap", "particle_effects", "shadow_effects"]:
        if not isinstance(config_data[key], bool):
            error_msg = f"{key} debe ser un valor booleano, se recibió {type(config_data[key])}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Verificar valores permitidos
    if config_data["visual_mode"] not in ["animated", "simple"]:
        error_msg = f"visual_mode debe ser 'animated' o 'simple', se recibió '{config_data['visual_mode']}'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuración validada correctamente")
    return True

def validate_model_file(model_path: str) -> bool:
    """
    Valida que un archivo de modelo exista y tenga el formato correcto.
    
    Args:
        model_path (str): Ruta al archivo del modelo
        
    Returns:
        bool: True si el archivo es válido
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo tiene un formato inválido
    """
    # Verificar que la ruta sea una cadena
    if not isinstance(model_path, str):
        error_msg = f"Ruta de modelo inválida: se esperaba una cadena, se recibió {type(model_path)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que el archivo existe
    if not os.path.exists(model_path):
        error_msg = f"El archivo de modelo no existe: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Verificar extensión del archivo
    if not model_path.endswith('.pth'):
        error_msg = f"Formato de archivo inválido: {model_path}. Se esperaba un archivo .pth"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que el archivo no esté vacío
    if os.path.getsize(model_path) == 0:
        error_msg = f"El archivo de modelo está vacío: {model_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Archivo de modelo validado correctamente: {model_path}")
    return True

def validate_model_content(checkpoint: Dict[str, Any]) -> bool:
    """
    Valida el contenido de un checkpoint de modelo cargado.
    
    Args:
        checkpoint (dict): Contenido del checkpoint
        
    Returns:
        bool: True si el contenido es válido
        
    Raises:
        ValueError: Si el contenido es inválido
    """
    # Verificar que checkpoint sea un diccionario
    if not isinstance(checkpoint, dict):
        error_msg = f"Checkpoint inválido: se esperaba un diccionario, se recibió {type(checkpoint)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que contenga la clave model_state_dict
    if "model_state_dict" not in checkpoint:
        error_msg = "Checkpoint inválido: falta la clave 'model_state_dict'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar tipos de datos para claves opcionales
    if "n_games" in checkpoint and not isinstance(checkpoint["n_games"], int):
        error_msg = f"n_games debe ser un entero, se recibió {type(checkpoint['n_games'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if "record" in checkpoint and not isinstance(checkpoint["record"], int):
        error_msg = f"record debe ser un entero, se recibió {type(checkpoint['record'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if "pathfinding_enabled" in checkpoint and not isinstance(checkpoint["pathfinding_enabled"], bool):
        error_msg = f"pathfinding_enabled debe ser un booleano, se recibió {type(checkpoint['pathfinding_enabled'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Contenido del checkpoint validado correctamente")
    return True

def validate_csv_file(csv_path: str) -> bool:
    """
    Valida que un archivo CSV exista y tenga el formato correcto.
    
    Args:
        csv_path (str): Ruta al archivo CSV
        
    Returns:
        bool: True si el archivo es válido
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo tiene un formato inválido
    """
    # Verificar que la ruta sea una cadena
    if not isinstance(csv_path, str):
        error_msg = f"Ruta de CSV inválida: se esperaba una cadena, se recibió {type(csv_path)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Si el archivo no existe, no es un error (puede ser la primera ejecución)
    if not os.path.exists(csv_path):
        logger.info(f"El archivo CSV no existe: {csv_path}. Se creará uno nuevo.")
        return True
    
    # Verificar extensión del archivo
    if not csv_path.endswith('.csv'):
        error_msg = f"Formato de archivo inválido: {csv_path}. Se esperaba un archivo .csv"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que el archivo no esté vacío
    if os.path.getsize(csv_path) == 0:
        logger.warning(f"El archivo CSV está vacío: {csv_path}")
        return True
    
    # Verificar que el archivo sea un CSV válido
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            # Leer al menos la primera fila para verificar formato
            header = next(reader, None)
            if header is None:
                logger.warning(f"El archivo CSV está vacío o mal formateado: {csv_path}")
                return True
    except Exception as e:
        error_msg = f"Error al leer el archivo CSV {csv_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Archivo CSV validado correctamente: {csv_path}")
    return True

def validate_csv_content(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Valida el contenido de un DataFrame cargado desde CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        required_columns (list): Lista de columnas requeridas
        
    Returns:
        bool: True si el contenido es válido
        
    Raises:
        ValueError: Si el contenido es inválido
    """
    # Verificar que df sea un DataFrame
    if not isinstance(df, pd.DataFrame):
        error_msg = f"Se esperaba un DataFrame, se recibió {type(df)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Si se especifican columnas requeridas, verificar que existan
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Faltan columnas requeridas en el DataFrame: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    logger.info("Contenido del DataFrame validado correctamente")
    return True

def validate_resource_file(file_path: str, allowed_extensions: List[str] = None) -> bool:
    """
    Valida que un archivo de recurso exista y tenga una extensión permitida.
    
    Args:
        file_path (str): Ruta al archivo de recurso
        allowed_extensions (list): Lista de extensiones permitidas
        
    Returns:
        bool: True si el archivo es válido
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo tiene una extensión no permitida
    """
    if allowed_extensions is None:
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.ttf', '.otf', '.wav', '.mp3']
    
    # Verificar que la ruta sea una cadena
    if not isinstance(file_path, str):
        error_msg = f"Ruta de recurso inválida: se esperaba una cadena, se recibió {type(file_path)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        error_msg = f"El archivo de recurso no existe: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Verificar extensión del archivo
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in allowed_extensions:
        error_msg = f"Extensión de archivo no permitida: {file_ext}. Extensiones permitidas: {allowed_extensions}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que el archivo no esté vacío
    if os.path.getsize(file_path) == 0:
        error_msg = f"El archivo de recurso está vacío: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Archivo de recurso validado correctamente: {file_path}")
    return True

def safe_json_load(file_path: str) -> Dict[str, Any]:
    """
    Carga un archivo JSON de forma segura con validación.
    
    Args:
        file_path (str): Ruta al archivo JSON
        
    Returns:
        dict: Contenido del archivo JSON
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el archivo no es un JSON válido
    """
    # Verificar que la ruta sea una cadena
    if not isinstance(file_path, str):
        error_msg = f"Ruta de JSON inválida: se esperaba una cadena, se recibió {type(file_path)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        error_msg = f"El archivo JSON no existe: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Verificar extensión del archivo
    if not file_path.endswith('.json'):
        error_msg = f"Formato de archivo inválido: {file_path}. Se esperaba un archivo .json"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Cargar el archivo JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Archivo JSON cargado correctamente: {file_path}")
        return data
    except json.JSONDecodeError as e:
        error_msg = f"El archivo no es un JSON válido: {file_path}. Error: {str(e)}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error al cargar el archivo JSON {file_path}: {str(e)}"
        logger.error(error_msg)
        raise

def safe_csv_load(file_path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV de forma segura con validación.
    
    Args:
        file_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Contenido del archivo CSV
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        pd.errors.EmptyDataError: Si el archivo está vacío
        pd.errors.ParserError: Si el archivo no es un CSV válido
    """
    # Validar el archivo
    validate_csv_file(file_path)
    
    # Cargar el archivo CSV
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Archivo CSV cargado correctamente: {file_path}")
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"El archivo CSV está vacío: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        error_msg = f"Error al cargar el archivo CSV {file_path}: {str(e)}"
        logger.error(error_msg)
        raise

def safe_model_load(model_path: str) -> Dict[str, Any]:
    """
    Carga un modelo de forma segura con validación.
    
    Args:
        model_path (str): Ruta al archivo del modelo
        
    Returns:
        dict: Contenido del checkpoint
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo tiene un formato inválido
        torch.serialization.pickle.UnpicklingError: Si el archivo no es un modelo PyTorch válido
    """
    # Validar el archivo
    validate_model_file(model_path)
    
    # Cargar el modelo
    try:
        checkpoint = torch.load(model_path)
        # Validar el contenido del checkpoint
        validate_model_content(checkpoint)
        logger.info(f"Modelo cargado correctamente: {model_path}")
        return checkpoint
    except torch.serialization.pickle.UnpicklingError:
        error_msg = f"El archivo no es un modelo PyTorch válido: {model_path}"
        logger.error(error_msg)
        raise
    except RuntimeError as e:
        error_msg = f"Error al cargar el modelo: {str(e)}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error inesperado al cargar el modelo: {str(e)}"
        logger.error(error_msg)
        raise
