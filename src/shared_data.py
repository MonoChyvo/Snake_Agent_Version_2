"""
Módulo para compartir datos entre diferentes componentes del programa.
Proporciona variables globales que pueden ser accedidas desde cualquier parte del código.
"""

# Parámetros del modelo compartidos
model_params = {
    "loss": 0.0,
    "temperature": 0.99,
    "learning_rate": 0.001,
    "pathfinding_enabled": True,
    "exploration_phase": False,
    "exploration_games_left": 0,
    "mode": "Pathfinding habilitado"
}

# Función para actualizar los parámetros del modelo
def update_model_params(params):
    """
    Actualiza los parámetros del modelo con los valores proporcionados.
    
    Args:
        params (dict): Diccionario con los nuevos valores para los parámetros.
    """
    global model_params
    
    # Actualizar los parámetros con los nuevos valores
    model_params.update(params)
    
    # Actualizar el modo basado en los parámetros
    if params.get("exploration_phase", False):
        model_params["mode"] = f"Exploración (restantes: {params.get('exploration_games_left', 0)})"
    elif params.get("pathfinding_enabled", False):
        model_params["mode"] = "Pathfinding habilitado"
    else:
        model_params["mode"] = "Explotación normal"
    
    # Imprimir para depuración
    print(f"[SHARED_DATA] Parámetros del modelo actualizados: {model_params}")

# Función para obtener los parámetros del modelo
def get_model_params():
    """
    Obtiene los parámetros actuales del modelo.
    
    Returns:
        dict: Diccionario con los parámetros del modelo.
    """
    return model_params.copy()
