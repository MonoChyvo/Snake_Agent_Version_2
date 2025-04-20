#!/usr/bin/env python
"""
Script para verificar el entorno de desarrollo.

Este script verifica que todas las dependencias necesarias estén instaladas
y que las versiones sean compatibles con el proyecto.
"""

import sys
import pkg_resources
import importlib
from tabulate import tabulate
import platform

def check_python_version():
    """Verifica la versión de Python."""
    required_version = (3, 7)
    current_version = sys.version_info

    if current_version < required_version:
        print(f"Error: Se requiere Python {required_version[0]}.{required_version[1]} o superior.")
        print(f"Versión actual: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False

    print(f"✓ Python {current_version[0]}.{current_version[1]}.{current_version[2]} (OK)")
    return True

def check_dependencies(requirements_file="requirements.txt"):
    """Verifica que todas las dependencias estén instaladas con las versiones correctas."""
    try:
        with open(requirements_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Filtrar líneas de comentarios y vacías
        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                # Extraer el nombre del paquete y la versión
                parts = line.split('#')[0].strip().split('==')
                if len(parts) == 2:
                    requirements.append((parts[0], parts[1]))

        # Verificar cada dependencia
        results = []
        all_ok = True

        for package, required_version in requirements:
            try:
                installed_version = pkg_resources.get_distribution(package).version
                status = "✓" if installed_version == required_version else "✗"
                if status == "✗":
                    all_ok = False
                results.append([package, required_version, installed_version, status])
            except pkg_resources.DistributionNotFound:
                results.append([package, required_version, "No instalado", "✗"])
                all_ok = False

        # Mostrar resultados
        print(f"\nVerificando dependencias en {requirements_file}:")
        print(tabulate(results, headers=["Paquete", "Versión requerida", "Versión instalada", "Estado"]))

        return all_ok

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {requirements_file}")
        return False
    except Exception as e:
        print(f"Error al verificar dependencias: {e}")
        return False

def check_gpu_support():
    """Verifica si hay soporte para GPU con PyTorch."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if has_cuda else 0
        device_name = torch.cuda.get_device_name(0) if has_cuda and device_count > 0 else "N/A"

        print("\nSoporte para GPU:")
        print(f"CUDA disponible: {'Sí' if has_cuda else 'No'}")
        print(f"Número de dispositivos: {device_count}")
        print(f"Dispositivo: {device_name}")

        return has_cuda
    except ImportError:
        print("\nNo se pudo verificar el soporte para GPU: PyTorch no está instalado")
        return False
    except Exception as e:
        print(f"\nError al verificar soporte para GPU: {e}")
        return False

def check_pygame():
    """Verifica la instalación de Pygame."""
    try:
        import pygame
        pygame_version = pygame.version.ver
        print(f"\n✓ Pygame {pygame_version} está instalado correctamente")

        # Intentar inicializar pygame para verificar que funciona
        pygame.init()
        drivers = pygame.display.get_driver()
        print(f"Driver de video: {drivers}")
        pygame.quit()

        return True
    except ImportError:
        print("\nError: Pygame no está instalado")
        return False
    except Exception as e:
        print(f"\nError al verificar Pygame: {e}")
        return False

def main():
    """Función principal."""
    print("=== Verificación del Entorno de Desarrollo ===\n")

    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Arquitectura: {platform.machine()}")
    print("")

    python_ok = check_python_version()
    deps_ok = check_dependencies()
    gpu_ok = check_gpu_support()
    pygame_ok = check_pygame()

    print("\n=== Resumen ===")
    print(f"Python: {'OK' if python_ok else 'ERROR'}")
    print(f"Dependencias: {'OK' if deps_ok else 'ERROR'}")
    print(f"Soporte GPU: {'OK' if gpu_ok else 'No disponible'}")
    print(f"Pygame: {'OK' if pygame_ok else 'ERROR'}")

    if python_ok and deps_ok and pygame_ok:
        print("\n✓ El entorno está configurado correctamente para el proyecto.")
        return 0
    else:
        print("\n✗ Hay problemas con el entorno. Revise los mensajes anteriores.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
