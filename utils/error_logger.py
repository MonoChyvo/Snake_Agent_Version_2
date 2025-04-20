"""
Módulo para el registro y manejo de errores en el sistema.
Proporciona funciones para registrar errores de manera consistente y detallada.
"""

import traceback
import logging
import os
from datetime import datetime

# Configurar el sistema de logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configurar el logger principal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"snake_dqn_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()  # También mostrar en consola
    ]
)

# Crear logger específico para errores del panel de estadísticas
stats_panel_logger = logging.getLogger('stats_panel')
stats_panel_logger.setLevel(logging.INFO)

# Crear logger específico para el sistema de eventos
event_system_logger = logging.getLogger('event_system')
event_system_logger.setLevel(logging.INFO)

# Crear logger específico para la actualización de datos
data_update_logger = logging.getLogger('data_update')
data_update_logger.setLevel(logging.INFO)

def log_error(logger, error_type, message, exception=None, context=None):
    """
    Registra un error con detalles completos.
    
    Args:
        logger: El logger a utilizar
        error_type: Tipo de error (ej: 'EventSystem', 'StatsPanel')
        message: Mensaje descriptivo del error
        exception: Excepción que causó el error (opcional)
        context: Información adicional sobre el contexto del error (opcional)
    """
    error_details = f"{error_type} ERROR: {message}"
    
    if context:
        error_details += f"\nContext: {context}"
    
    if exception:
        error_details += f"\nException: {str(exception)}"
        error_details += f"\nTraceback: {traceback.format_exc()}"
    
    logger.error(error_details)
    
    # También imprimir en consola para depuración inmediata
    print(f"[ERROR] {error_type}: {message}")

def log_warning(logger, warning_type, message, context=None):
    """
    Registra una advertencia.
    
    Args:
        logger: El logger a utilizar
        warning_type: Tipo de advertencia
        message: Mensaje descriptivo
        context: Información adicional sobre el contexto (opcional)
    """
    warning_details = f"{warning_type} WARNING: {message}"
    
    if context:
        warning_details += f"\nContext: {context}"
    
    logger.warning(warning_details)
    
    # También imprimir en consola para depuración inmediata
    print(f"[WARNING] {warning_type}: {message}")

def log_info(logger, info_type, message, context=None):
    """
    Registra información.
    
    Args:
        logger: El logger a utilizar
        info_type: Tipo de información
        message: Mensaje descriptivo
        context: Información adicional sobre el contexto (opcional)
    """
    info_details = f"{info_type}: {message}"
    
    if context:
        info_details += f"\nContext: {context}"
    
    logger.info(info_details)
