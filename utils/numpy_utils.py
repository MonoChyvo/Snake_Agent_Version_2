"""
Utilidades para operaciones seguras con NumPy.
Proporciona funciones que evitan warnings comunes.
"""

import numpy as np
import warnings

def safe_mean(arr, default=0.0):
    """
    Calcula la media de un array de forma segura, evitando warnings.
    
    Args:
        arr: Array o lista de valores
        default: Valor por defecto si el array está vacío
        
    Returns:
        float: Media del array o valor por defecto
    """
    # Convertir a numpy array si no lo es
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    # Verificar si el array está vacío
    if arr.size == 0:
        return default
    
    # Calcular la media de forma segura
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.mean(arr)
    
    # Verificar si el resultado es NaN o infinito
    if np.isnan(result) or np.isinf(result):
        return default
    
    return result

def safe_std(arr, default=1.0):
    """
    Calcula la desviación estándar de un array de forma segura, evitando warnings.
    
    Args:
        arr: Array o lista de valores
        default: Valor por defecto si el array está vacío o tiene un solo elemento
        
    Returns:
        float: Desviación estándar del array o valor por defecto
    """
    # Convertir a numpy array si no lo es
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    # Verificar si el array está vacío o tiene un solo elemento
    if arr.size <= 1:
        return default
    
    # Calcular la desviación estándar de forma segura
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.std(arr)
    
    # Verificar si el resultado es NaN o infinito
    if np.isnan(result) or np.isinf(result):
        return default
    
    # Evitar valores muy pequeños que podrían causar problemas numéricos
    return max(result, 1e-8)

def safe_normalize(arr, mean=None, std=None):
    """
    Normaliza un array de forma segura (z-score normalization).
    
    Args:
        arr: Array a normalizar
        mean: Media a usar para normalizar (si es None, se calcula)
        std: Desviación estándar a usar (si es None, se calcula)
        
    Returns:
        numpy.ndarray: Array normalizado
    """
    # Convertir a numpy array si no lo es
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    # Si el array está vacío, devolver array vacío
    if arr.size == 0:
        return arr
    
    # Calcular media y desviación estándar si no se proporcionan
    if mean is None:
        mean = safe_mean(arr)
    if std is None:
        std = safe_std(arr)
    
    # Normalizar
    return (arr - mean) / std
