# Manual de Usuario

## Instalación

1. Asegúrate de tener Python 3.7 o superior instalado
2. Clona el repositorio:
   ```
   git clone https://github.com/tu-usuario/snake-dqn.git
   cd snake-dqn
   ```
3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Ejecución

Para iniciar el entrenamiento del agente:

```
python main.py
```

## Controles

Durante la ejecución, puedes utilizar los siguientes controles:

- **V**: Cambiar entre modo visual animado y simple
- **G**: Activar/desactivar visualización de la cuadrícula
- **H**: Activar/desactivar visualización del mapa de calor
- **P**: Activar/desactivar pathfinding
- **S**: Cambiar tamaño de la ventana
- **ESC**: Salir del juego

## Modos de Visualización

### Modo Animado

El modo animado ofrece una experiencia visual más rica con:
- Efectos de partículas al comer comida
- Sombras y efectos de movimiento para la serpiente
- Transiciones suaves entre estados

### Modo Simple

El modo simple proporciona una visualización más básica y eficiente:
- Renderizado simple de la serpiente y la comida
- Sin efectos visuales adicionales
- Mayor rendimiento en equipos con recursos limitados

## Configuración

Puedes personalizar diversos aspectos del juego y del entrenamiento modificando el archivo `utils/config.py`:

### Parámetros del Juego

- `BLOCK_SIZE`: Tamaño de cada bloque en píxeles
- `SPEED`: Velocidad del juego
- `MAX_EPOCHS`: Número máximo de épocas de entrenamiento

### Configuración Visual

- `VISUAL_MODE`: Modo de visualización ("animated" o "simple")
- `SHOW_GRID`: Mostrar cuadrícula en el fondo
- `SHOW_HEATMAP`: Mostrar mapa de calor de posiciones visitadas
- `PARTICLE_EFFECTS`: Activar efectos de partículas
- `SHADOW_EFFECTS`: Activar efectos de sombra

### Hiperparámetros de Entrenamiento

- `LR`: Tasa de aprendizaje
- `GAMMA`: Factor de descuento
- `BATCH_SIZE`: Tamaño del lote
- `TEMPERATURE`: Temperatura para exploración

## Interpretación de la Interfaz

La interfaz del juego muestra la siguiente información:

- **Score**: Puntuación actual (número de comidas recolectadas)
- **Game**: Número de juego actual
- **Record**: Puntuación más alta alcanzada

El área principal muestra:
- La serpiente (en azul/verde)
- La comida (en rojo)
- La cuadrícula (si está activada)
- El mapa de calor (si está activado)

## Solución de Problemas

### El juego se ejecuta lentamente

- Cambia al modo de visualización simple (tecla V)
- Desactiva el mapa de calor (tecla H)
- Desactiva la cuadrícula (tecla G)

### Errores de importación

- Asegúrate de ejecutar el juego desde el directorio raíz del proyecto
- Verifica que todas las dependencias estén instaladas correctamente

### Otros problemas

Si encuentras otros problemas, consulta la sección de issues en el repositorio o crea un nuevo issue describiendo el problema en detalle.
