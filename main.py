"""
Punto de entrada principal para el proyecto Snake DQN.

Este archivo inicia el entrenamiento del agente con la configuración visual
seleccionada por el usuario. Gestiona la inicialización del entorno y el manejo
de excepciones durante la ejecución.

Versión: 1.0.0
"""

import sys
import pygame
from src.agent import train
from utils.config import MAX_EPOCHS

def main():
    """
    Función principal que inicia el entrenamiento del agente.

    Inicializa pygame, ejecuta el entrenamiento y maneja posibles excepciones
    para asegurar un cierre limpio del programa.
    """
    # Inicializar pygame
    pygame.init()

    try:
        # Iniciar entrenamiento con el número máximo de épocas
        train(MAX_EPOCHS)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    except Exception as e:
        import traceback
        print(f"Error durante el entrenamiento: {e}")
        print("Detalles del error:")
        traceback.print_exc()
    finally:
        # Asegurar que pygame se cierre correctamente
        pygame.quit()
        print("Programa finalizado.")

if __name__ == "__main__":
    main()
