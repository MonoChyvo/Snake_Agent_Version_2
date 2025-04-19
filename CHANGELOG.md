# Registro de Cambios

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Versionado Semántico](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2023-12-15

### Añadido
- Implementación inicial del juego Snake con PyGame
- Agente de aprendizaje por refuerzo basado en DQN
- Visualización en tiempo real del proceso de entrenamiento
- Interfaz gráfica con diseño de "estadio"
- Modos de visualización: animado y simple
- Sistema de pathfinding avanzado
- Análisis y seguimiento de métricas de entrenamiento
- Documentación completa del proyecto

### Cambiado
- Reorganización de la estructura del proyecto en directorios src/, utils/ y docs/
- Mejora de la documentación con docstrings detallados
- Actualización del archivo .gitignore para una mejor gestión de archivos

### Corregido
- Ajuste de la cuadrícula para eliminar el excedente en los bordes
- Optimización de la generación de comida y serpiente
- Perfeccionamiento de la detección de colisiones

## [0.9.0] - 2023-11-30

### Añadido
- Versión preliminar del juego Snake
- Implementación básica del algoritmo DQN
- Interfaz gráfica simple
- Sistema básico de recompensas

### Cambiado
- Mejora del rendimiento de visualización en tiempo real
- Optimización del algoritmo de aprendizaje

### Corregido
- Corrección de errores en la detección de colisiones
- Solución de problemas de memoria durante el entrenamiento prolongado
