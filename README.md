# Snake DQN - Juego de Snake con Aprendizaje por Refuerzo Profundo

<p align="center">
  <img src="assets/snake_dqn_logo.png" alt="Snake DQN Logo" width="200" height="200" />
</p>

Este proyecto implementa el clásico juego de Snake utilizando Deep Q-Learning (DQN) para entrenar un agente que aprenda a jugar de forma autónoma. El sistema utiliza técnicas avanzadas de aprendizaje por refuerzo para desarrollar estrategias óptimas de juego.

## 💫 Características Principales

- **Implementación Completa del Juego**: Desarrollado con PyGame, con múltiples modos de visualización.
- **Algoritmo DQN Avanzado**: Incluye mejoras como Double DQN, Prioritized Experience Replay y estrategias de exploración adaptativas.
- **Visualización en Tiempo Real**: Observa cómo el agente aprende y mejora con el tiempo.
- **Interfaz Gráfica Mejorada**: Diseño de "estadio" con efectos visuales personalizables.
- **Sistema de Pathfinding**: Algoritmos A\* y búsqueda de caminos largos para evitar situaciones de bloqueo.
- **Análisis Detallado**: Seguimiento y visualización de métricas de entrenamiento.
- **Configuración Persistente**: Guarda tus preferencias visuales entre sesiones.

## 💻 Requisitos Técnicos

- Python 3.7 o superior
- Dependencias principales:
  - PyGame 2.6.1 (motor del juego)
  - PyTorch 1.13.1 (framework de aprendizaje profundo)
  - NumPy 1.21.6 (procesamiento numérico)
  - Pandas 1.3.5 (análisis de datos)
  - Matplotlib 3.5.3 y Seaborn 0.12.2 (visualización)
  - Colorama 0.4.4 (salida de consola coloreada)

## 📍 Instalación

### Opción 1: Usando Scripts de Configuración (Recomendado)

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/snake-dqn.git
   cd snake-dqn
   ```

2. Ejecuta el script de configuración:

   - En Windows:
     ```bash
     setup_env.bat
     ```
   - En macOS/Linux:
     ```bash
     chmod +x setup_env.sh
     ./setup_env.sh
     ```

   Estos scripts crearán un entorno virtual, lo activarán e instalarán todas las dependencias necesarias.

### Opción 2: Instalación Manual

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/snake-dqn.git
   cd snake-dqn
   ```

2. Crea y activa un entorno virtual (recomendado):

   - En Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

   El archivo requirements.txt incluye todas las dependencias necesarias, con comentarios claros sobre cuáles son esenciales y cuáles son opcionales. Puedes editar este archivo para descomentar las dependencias adicionales que necesites.

### Verificación del Entorno

Para verificar que tu entorno está configurado correctamente:

```bash
python check_env.py
```

Este script comprobará que todas las dependencias estén instaladas con las versiones correctas y que el sistema esté listo para ejecutar el proyecto.

## 🎮 Uso

### Iniciar el Entrenamiento

Para comenzar el entrenamiento del agente:

```bash
python main.py
```

Al iniciar, se mostrará una pantalla de configuración donde puedes seleccionar las opciones visuales:

- **Modo Animado**: Visualización completa con efectos gráficos.
- **Modo Simple**: Renderizado básico para mayor rendimiento.

### Controles Durante el Entrenamiento

- **V**: Alterna entre modo visual animado y simple.
- **P**: Activa/desactiva el sistema de pathfinding.
- **S**: Cambia el tamaño de la ventana (pequeño, mediano, grande).

### Limpiar Archivos Temporales

Para eliminar archivos de caché y temporales:

```bash
python clean.py
```

## 📊 Seguimiento del Entrenamiento

Durante el entrenamiento, el sistema genera:

- **Gráficos de Progreso**: Visualización de puntuaciones y recompensas.
- **Archivos CSV**: Datos detallados para análisis posterior.
- **Modelos Guardados**: Puntos de control del entrenamiento.

Los resultados se guardan en las carpetas `plots/` y `results/`.

## 📂 Estructura del Proyecto

```
snake_dqn/
├── assets/            # Recursos gráficos y fuentes
├── docs/              # Documentación adicional
├── model_Model/       # Modelos guardados
├── plots/             # Gráficos generados
├── results/           # Resultados y análisis
├── src/               # Código fuente principal
│   ├── agent.py       # Agente de aprendizaje
│   ├── game.py        # Implementación del juego
│   ├── model.py       # Arquitectura de la red neuronal
│   └── start_screen.py # Pantalla de inicio
├── utils/             # Utilidades y herramientas
│   ├── advanced_pathfinding.py  # Algoritmos de búsqueda
│   ├── config.py                # Configuración y parámetros
│   ├── config_manager.py        # Gestión de configuración
│   ├── efficient_memory.py      # Gestión optimizada de memoria
│   ├── evaluation.py            # Herramientas de evaluación
│   └── helper.py                # Funciones auxiliares
├── clean.py           # Script para limpiar archivos temporales
├── inspection.py       # Herramientas de inspección y análisis
├── main.py            # Punto de entrada principal
├── requirements.txt    # Dependencias del proyecto
└── README.md          # Este archivo
```

## ⚙️ Configuración

### Configuración Visual

La configuración visual se puede ajustar a través de la pantalla de inicio y se guarda en `config.json` para futuras sesiones.

### Parámetros del Sistema

Puedes modificar los parámetros del juego y del entrenamiento en `utils/config.py`:

- **Parámetros de Juego**: Tamaño de bloque, velocidad, dimensiones.
- **Parámetros Visuales**: Efectos, colores, animaciones.
- **Hiperparámetros de DQN**: Tasa de aprendizaje, factor de descuento, tamaño de lote.
- **Parámetros de Exploración**: Temperatura, decaimiento, fases de exploración.
- **Gestión de Memoria**: Tamaño del búfer, umbrales de poda.

## 💡 Algoritmos Implementados

### Deep Q-Network (DQN)

El proyecto implementa varias mejoras sobre el algoritmo DQN básico:

- **Double DQN**: Reduce la sobreestimación de valores Q utilizando una red objetivo.
- **Prioritized Experience Replay**: Muestrea experiencias más importantes con mayor frecuencia.
- **Exploración Adaptativa**: Ajusta dinámicamente la exploración según el progreso.

### Pathfinding

- **Algoritmo A\***: Encuentra el camino más corto hacia la comida.
- **Búsqueda de Caminos Largos**: Evita situaciones de bloqueo en serpientes largas.
- **Análisis de Espacio Libre**: Toma decisiones estratégicas basadas en el espacio disponible.

## 📈 Rendimiento

El sistema incluye herramientas para monitorear y analizar el rendimiento:

- **Seguimiento de Métricas**: Puntuación, longitud, recompensa, pérdida.
- **Alertas Automáticas**: Detecta problemas potenciales durante el entrenamiento.
- **Evaluación Periódica**: Prueba el rendimiento del agente en escenarios controlados.

## 📝 Documentación Adicional

Para más detalles sobre la arquitectura y el diseño del sistema, consulta los archivos en la carpeta `docs/`:

- [Arquitectura del Sistema](docs/architecture.md)
- [Manual de Usuario](docs/user_manual.md)

## 💬 Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir:

1. Haz un fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`)
4. Sube los cambios a tu fork (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 🔒 Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

## 👨‍💻 Autor

Desarrollado como parte de un proyecto de investigación en aprendizaje por refuerzo.

---

<p align="center">
  <i>"La inteligencia artificial es la nueva electricidad." - Andrew Ng</i>
</p>
