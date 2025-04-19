# Documentación del Proyecto Snake DQN

Esta carpeta contiene la documentación detallada del proyecto Snake DQN.

## Contenido

- [Arquitectura del Proyecto](architecture.md)
- [Guía de Instalación](installation.md)
- [Manual de Usuario](user_manual.md)
- [Guía para Desarrolladores](developer_guide.md)
- [Referencia de la API](api_reference.md)

## Descripción General

Snake DQN es una implementación del clásico juego Snake que utiliza aprendizaje por refuerzo profundo (Deep Q-Learning) para entrenar un agente que aprenda a jugar de forma autónoma.

El proyecto está estructurado en módulos que separan claramente las responsabilidades:
- El entorno del juego (src/game.py)
- El agente de aprendizaje (src/agent.py)
- La arquitectura de la red neuronal (src/model.py)
- Utilidades y configuración (utils/)

## Características Principales

- Implementación completa del juego Snake con PyGame
- Agente de aprendizaje por refuerzo basado en DQN
- Visualización en tiempo real del proceso de entrenamiento
- Interfaz gráfica mejorada con diseño de "estadio"
- Múltiples modos de visualización
- Sistema avanzado de pathfinding
- Análisis y seguimiento de métricas de entrenamiento
