# Arquitectura del Proyecto

## Estructura de Directorios

```
snake_dqn/
├── assets/              # Recursos gráficos y fuentes
├── docs/                # Documentación detallada
├── src/                 # Código fuente principal
│   ├── __init__.py
│   ├── agent.py         # Agente de aprendizaje por refuerzo
│   ├── game.py          # Implementación del juego Snake
│   ├── model.py         # Arquitectura de la red neuronal
│   └── ...
├── utils/               # Utilidades y herramientas
│   ├── __init__.py
│   ├── advanced_pathfinding.py  # Algoritmos de búsqueda de caminos
│   ├── config.py        # Configuración y parámetros
│   └── ...
├── main.py              # Punto de entrada principal
├── requirements.txt     # Dependencias del proyecto
├── LICENSE              # Licencia del proyecto
└── README.md            # Documentación general
```

## Componentes Principales

### Entorno del Juego (game.py)

El entorno del juego implementa la mecánica del juego Snake y proporciona una interfaz para que el agente interactúe con él. Características principales:

- Renderizado visual con PyGame
- Sistema de recompensas para el aprendizaje por refuerzo
- Detección de colisiones y gestión de estados
- Múltiples modos de visualización

### Agente de Aprendizaje (agent.py)

El agente implementa el algoritmo Deep Q-Learning (DQN) para aprender a jugar al Snake. Características principales:

- Implementación de DQN con experiencia de repetición
- Exploración vs. explotación con política epsilon-greedy
- Red neuronal para aproximar la función Q
- Actualización de pesos con descenso de gradiente

### Modelo de Red Neuronal (model.py)

Define la arquitectura de la red neuronal utilizada por el agente. Características principales:

- Red neuronal profunda con capas totalmente conectadas
- Función de activación ReLU
- Capa de salida lineal para estimar valores Q

### Configuración (config.py)

Centraliza todos los parámetros y constantes utilizados en el proyecto:

- Hiperparámetros de entrenamiento
- Configuración visual
- Parámetros del juego
- Umbrales de alerta para monitoreo

### Pathfinding Avanzado (advanced_pathfinding.py)

Implementa algoritmos de búsqueda de caminos para ayudar al agente:

- Algoritmo A* para encontrar el camino más corto
- Búsqueda de caminos largos para evitar quedar encerrado
- Análisis de espacio libre para toma de decisiones estratégicas

## Flujo de Ejecución

1. El usuario inicia el programa a través de `main.py`
2. Se crea una instancia del entorno del juego (`SnakeGameAI`)
3. Se inicializa el agente con la red neuronal
4. El agente interactúa con el entorno en un bucle:
   - Observa el estado actual
   - Selecciona una acción según su política
   - Ejecuta la acción en el entorno
   - Recibe una recompensa y observa el nuevo estado
   - Actualiza su política basándose en la experiencia
5. El proceso continúa hasta alcanzar el número máximo de épocas o hasta que el usuario interrumpe el entrenamiento
