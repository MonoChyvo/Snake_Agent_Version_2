# Análisis del Proyecto Snake DQN

Este documento detalla el análisis del proyecto de Agente de Inteligencia Artificial para el juego Snake utilizando Deep Q-Learning (DQN).

## 1. Visión General
El proyecto es una implementación completa de un agente de Aprendizaje por Refuerzo (Reinforcement Learning) entrenado para jugar Snake. Utiliza Pygame para la visualización y PyTorch para el modelo de red neuronal.

## 2. Arquitectura del Sistema

### Entrada (`main.py`)
- Punto de entrada simple.
- Inicializa Pygame y lanza el bucle de entrenamiento (`train` en `src/agent.py`).
- Manejo de excepciones básico para cierre limpio.

### Modelo (`src/model.py`)
- **DQN (Deep Q-Network)**: Red neuronal de 3 capas (Input -> Hidden -> Hidden -> Output).
- **QTrainer**:
  - Implementa **Double DQN** (usa `target_model` para estabilizar el entrenamiento).
  - **Regularización L2**: Penalización específica por capa para evitar sobreajuste.
  - Optimizador Adam.
  - Función de pérdida MSE (Mean Squared Error).

### Agente (`src/agent.py`)
- **Cerebro del sistema**.
- **Memoria**: Usa `EfficientPrioritizedReplayMemory` (Memoria de Repetición Priorizada) para aprender más de experiencias importantes.
- **Exploración vs Explotación**:
  - Sistema dinámico basado en temperatura (similar a Simulated Annealing/Softmax) en lugar de Epsilon-Greedy tradicional.
  - Fases de exploración forzada periódicas.
- **Entrenamiento**:
  - `train_short_memory`: Entrena paso a paso.
  - `train_long_memory`: Entrena con un lote de experiencias (replay buffer).
- **Estado**: Representación compleja de 23 dimensiones (peligros, dirección, comida, densidad, distancias, etc.).

### Entorno (`src/game/core.py`)
- **SnakeGameAI**: Maneja la lógica del juego.
- **Recompensas**: Sistema sofisticado que premia:
  - Comer comida (+10 max).
  - Acercarse a la comida (función no lineal).
  - Sobrevivir.
  - Espacio libre (evitar encerrarse).
  - Penaliza colisiones y movimientos ineficientes (bucles).
- **Visualización**:
  - UI rica con Pygame.
  - Mapa de calor de visitas.
  - Partículas y efectos visuales.
  - Panel de estadísticas en tiempo real.

## 3. Puntos Fuertes Detectados
1. **Técnicas Avanzadas de RL**: No es un DQN básico. Implementa Double DQN y Prioritized Replay, lo que mejora significativamente la estabilidad y velocidad de convergencia.
2. **Ingeniería de Recompensas**: El sistema de recompensas es granular y tiene en cuenta múltiples factores (futuro, espacio, distancia), lo cual ayuda al agente a aprender estrategias más complejas que solo "ir a la comida".
3. **Arquitectura Modular**: Buena separación de responsabilidades entre Modelo, Agente y Juego.
4. **Experiencia de Usuario (UX)**: La visualización es superior al promedio para proyectos de este tipo, con gráficos informativos y opciones de configuración.
5. **Robustez**: Manejo de excepciones en carga de modelos, assets y bucle principal.

## 4. Áreas de Mejora Potencial
1. **Hardcoded Paths**: Se observan rutas como `assets/arial.ttf` o `./model_Model` que podrían beneficiarse de una configuración centralizada de rutas absolutas/relativas para evitar errores si se ejecuta desde otro directorio.
2. **Complejidad del Estado**: El estado de 23 dimensiones es rico, pero podría revisarse si hay redundancia (ej. correlación entre densidades y distancias).
3. **Magic Numbers**: Hay varios valores numéricos (factores de recompensa, tamaños de grid) dispersos en el código que podrían centralizarse más en `config.py` o `config.json`.
4. **Logica de Pathfinding**: Se menciona `AdvancedPathfinding` pero el agente a veces mezcla lógica basada en reglas (pathfinding) con predicción neuronal, lo cual puede ser confuso de depurar si el agente hace algo "extraño" (¿fue la red o el pathfinding?).

## 5. Conclusión
Es un proyecto sólido y avanzado. Está más cerca de una aplicación pulida que de un script de prueba básico.