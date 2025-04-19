#!/usr/bin/env python
"""
Script para limpiar archivos temporales y de caché del proyecto.

Este script elimina archivos temporales y de caché generados durante
el desarrollo y ejecución del proyecto, como archivos __pycache__,
archivos .pyc, y otros archivos temporales.

Uso:
    python clean.py

Versión: 1.0.0
"""

import os
import shutil
import fnmatch

def clean_project():
    """
    Limpia archivos temporales y de caché del proyecto.
    
    Elimina:
    - Directorios __pycache__
    - Archivos .pyc, .pyo, .pyd
    - Archivos .DS_Store
    - Directorios de caché de pytest
    """
    print("Limpiando archivos temporales y de caché...")
    
    # Directorios a eliminar
    dirs_to_remove = [
        "__pycache__",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist",
        "*.egg-info",
    ]
    
    # Archivos a eliminar
    files_to_remove = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.c",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        ".coverage",
        "*.bak",
        "*.tmp",
        "*.swp",
    ]
    
    # Obtener el directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Eliminar directorios
    for root, dirs, files in os.walk(current_dir):
        for pattern in dirs_to_remove:
            for directory in fnmatch.filter(dirs, pattern):
                path = os.path.join(root, directory)
                print(f"Eliminando directorio: {path}")
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    print(f"Error al eliminar {path}: {e}")
    
    # Eliminar archivos
    for root, dirs, files in os.walk(current_dir):
        for pattern in files_to_remove:
            for filename in fnmatch.filter(files, pattern):
                path = os.path.join(root, filename)
                print(f"Eliminando archivo: {path}")
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error al eliminar {path}: {e}")
    
    print("Limpieza completada.")

if __name__ == "__main__":
    clean_project()
