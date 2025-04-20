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
- **Seguridad Mejorada**: Validación robusta de datos y manejo avanzado de excepciones.

## 💻 Requisitos Técnicos

- Python 3.7 o superior
- Dependencias principales:
  - PyGame >= 2.0.0 (motor del juego)
  - PyTorch >= 1.7.0 (framework de aprendizaje profundo)
  - NumPy >= 1.19.0 (procesamiento numérico)
  - Pandas >= 1.1.0 (análisis de datos)
  - Matplotlib >= 3.3.0 y Seaborn >= 0.11.0 (visualización)

## 📍 Instalación

### Opción 1: Instalación Estándar

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/snake-dqn.git
   cd snake-dqn
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Opción 2: Usando un Entorno Virtual (Recomendado)

1. Clona el repositorio y crea un entorno virtual:

   ```bash
   git clone https://github.com/tu-usuario/snake-dqn.git
   cd snake-dqn
   python -m venv venv
   ```

2. Activa el entorno virtual:

   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

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
│   ├── helper.py                # Funciones auxiliares
│   └── validation.py            # Validación de datos y seguridad
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
- [Mejoras de Seguridad](docs/security_improvements.md)
- [Implementación de Seguridad](docs/security_implementation.md)

## 💬 Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir:

1. Haz un fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`)
4. Sube los cambios a tu fork (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 🔐 Seguridad

El proyecto implementa varias capas de seguridad para garantizar la robustez y estabilidad:

- **Validación de Entrada**: Verificación exhaustiva de todos los datos externos.
- **Manejo de Excepciones**: Recuperación elegante de errores en áreas críticas.
- **Registro de Seguridad**: Seguimiento detallado de eventos y errores.
- **Validación de Recursos**: Verificación de integridad de archivos cargados.
- **Protección contra Datos Malformados**: Prevención de fallos por datos corruptos.

Para más detalles, consulta la [documentación de seguridad](docs/security_implementation.md).

## 🔒 Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

## 👨‍💻 Autor

Desarrollado como parte de un proyecto de investigación en aprendizaje por refuerzo.

---

<p align="center">
  <i>"La inteligencia artificial es la nueva electricidad." - Andrew Ng</i>
</p>
