# Snake DQN - Juego de Snake con Aprendizaje por Refuerzo Profundo

Este proyecto implementa el clásico juego de Snake utilizando Deep Q-Learning (DQN) para entrenar un agente que aprende a jugar de forma autónoma. El sistema ahora solo incluye la versión animada, eliminando modos visuales alternativos, pantalla de inicio y configuraciones visuales avanzadas.

## 💫 Características Principales

- **Versión Animada Única**: El juego siempre inicia en modo animado, sin selección de modos ni configuraciones visuales adicionales.
- **Interfaz Gráfica Limpia**: Diseño simplificado, solo con los elementos necesarios para el entrenamiento y visualización animada.
- **Sistema de Pathfinding**: Algoritmos A* y búsqueda de caminos largos para evitar bloqueos.
- **Análisis Detallado**: Seguimiento y visualización de métricas de entrenamiento.

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

### Controles Durante el Entrenamiento

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
├── assets/            # Recursos visuales
│   ├── apple.png      # Imagen de la manzana
│   ├── arial.ttf      # Fuente usada en la interfaz
├── src/               # Código fuente principal
│   ├── agent.py       # Agente de aprendizaje
│   ├── game.py        # Implementación del juego (solo animado)
│   ├── model.py       # Arquitectura de la red neuronal
│   ├── shared_data.py # Parámetros compartidos
│   ├── stats_manager.py # Gestión centralizada de estadísticas
├── utils/             # Utilidades y herramientas
│   ├── config.py      # Configuración general (solo animación)
│   ├── advanced_pathfinding.py  # Algoritmos de búsqueda
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

### Parámetros del Sistema

Puedes modificar los parámetros del juego y del entrenamiento en `utils/config.py`:

- **Parámetros de Juego**: Tamaño de bloque, velocidad, dimensiones.
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

## 📂 Panel de Estadísticas: Integración, Cobertura y Pruebas

### Arquitectura y Funcionamiento

El **panel de estadísticas** está gestionado por la clase `StatsManager`, que centraliza la recolección, actualización y notificación de cambios en las métricas del juego y del agente. El panel se refresca de manera eficiente gracias a un sistema de eventos y un dirty flag, asegurando que solo se actualice cuando hay cambios reales en los datos.

#### Métricas cubiertas:
- **Básicas:** Puntuación, Récord, Pasos
- **Eficiencia:** Ratio de eficiencia, Pasos por comida
- **Acciones:** Recto %, Derecha %, Izquierda %
- **Entrenamiento:** Recompensa media, Último récord (juego)
- **Modelo:** Pérdida, Temperatura, Learning rate, Pathfinding, Modo de explotación

#### Flujo de integración
1. El juego y el agente actualizan sus métricas internas.
2. `StatsManager` detecta cualquier cambio relevante (comparación profunda por categoría).
3. Si hay cambios, activa el dirty flag y notifica a la UI mediante el sistema de eventos.
4. El panel de estadísticas se refresca solo cuando el dirty flag está activo, mostrando siempre la información más reciente y precisa.

### Pruebas Unitarias y de Integración

El archivo `test_stats_event_system.py` incluye **tests exhaustivos** para cada grupo de métricas y para la integración completa del panel:
- Cada test verifica que el valor mostrado en el panel corresponde al valor actualizado en el juego o el agente.
- Se comprueba que el dirty flag y la notificación de eventos funcionan correctamente.
- El test de integración simula el ciclo completo: actualización de métrica, refresco del panel y verificación de la visualización.

#### Ejecución de las pruebas

Para validar que todo el panel y el sistema de eventos funcionan correctamente:

```bash
python -m unittest test_stats_event_system.py
```

Si todos los tests pasan (`OK`), puedes estar seguro de que la integración entre el panel, el sistema de eventos y el backend es robusta y funcional.

### Validación manual en la interfaz

1. Ejecuta el juego normalmente:
   ```bash
   python main.py
   ```
2. Observa el panel de estadísticas: cada vez que cambies una métrica (por ejemplo, al superar un récord), el valor debe actualizarse automáticamente y sin retrasos.
3. Si detectas un valor incorrecto, ejecuta nuevamente los tests para aislar el problema.

---

**¡Con esta arquitectura y cobertura de pruebas, puedes confiar en la precisión y eficiencia del panel de estadísticas, tanto a nivel interno como visual!**

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
