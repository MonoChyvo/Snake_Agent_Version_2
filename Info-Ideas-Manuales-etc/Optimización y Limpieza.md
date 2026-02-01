# Informe de Optimización y Limpieza

Este informe detalla áreas específicas para reducir la complejidad, eliminar código innecesario y mejorar el rendimiento visual del proyecto Snake DQN.

## 1. Código Innecesario y Limpieza General

### Archivos Redundantes o Bloat
- `utils/config.py` y `utils/config_manager.py`: Parece haber duplicidad en el manejo de configuración. Se recomienda unificar todo en una sola clase o módulo de configuración para evitar confusiones sobre cuál es la "fuente de la verdad".
- `utils/numpy_utils.py` y `utils/helper.py`: Revisar si las funciones en `numpy_utils` son tan específicas que merecen su propio archivo o si pueden integrarse en un módulo de utilidades generales de matemáticas.
- `__pycache__`: Asegurarse de que `.gitignore` esté configurado correctamente para no rastrear estos archivos (ya parece estarlo, pero siempre es bueno verificar).

### Limpieza de Código
- **Magic Numbers**: Muchos valores numéricos (colores, tamaños de margen, offsets de sombra) están dispersos en `core.py`. Deberían moverse a `config.py` o constantes globales para una gestión más limpia.
- **Comentarios obsoletos**: Revisar comentarios como `# -- NUEVO --` que podrían ya no ser relevantes si el código lleva tiempo integrado.

## 2. Elementos Innecesarios para Rendimiento Visual

Si el objetivo es maximizar FPS o reducir el uso de CPU/GPU, se pueden deshabilitar o optimizar los siguientes elementos:

### Cuello de Botella Principal: Mapas de Calor (`_draw_heatmap`)
- **Problema**: El método `_draw_heatmap` se llama en cada frame (`_update_ui` -> `_draw_heatmap`). Redibuja TODOS los bloques del grid, calculando colores y transparencias cada vez.
- **Impacto**: Alto costo computacional `O(W*H)` por frame, donde W y H son dimensiones del grid.
- **Solución**:
  - Actualizar el mapa de calor solo cuando la serpiente se mueve, no en cada frame de renderizado.
  - O mejor aún, dibujar sobre una `Surface` persistente y solo hacer `blit` de esa superficie, actualizando solo el píxel/bloque que cambió.

### Partículas y Efectos (`_render_animated`)
- **Problema**: Itera sobre una lista de partículas para actualizarlas y dibujarlas una por una.
- **Impacto**: Puede ralentizarse si hay muchas partículas activas (ej. al comer mucha comida rápido).
- **Mejora**: Usar renderizado instanciado o limitar el número máximo de partículas. O simplemente desactivarlas si se prefiere rendimiento puro.

### Sombras y Bordes Redondeados
- **Problema**: `pygame.draw.rect` con `border_radius` y el dibujo duplicado para sombras (dibujar rect negro + rect color) duplica las llamadas de dibujo por segmento de serpiente.
- **Impacto**: Lineal respecto a la longitud de la serpiente `O(L)`.
- **Mejora**: Para entrenamientos muy largos donde la serpiente es gigante, esto se notará. Se puede simplificar a rectángulos simples sin sombras ni bordes redondeados.

### Fuentes y Texto
- **Problema**: Renderizar texto dinámico (scores, stats) en cada frame puede ser costoso si no se cachea.
- **Mejora**: Pygame es lento renderizando texto. Asegurarse de que las superficies de texto solo se regeneren cuando el valor cambia (el código actual parece intentar esto con `stats_panel_needs_update`, lo cual es bueno).

## 3. Recomendaciones de Acción Inmediata

1. **Refactorizar `_draw_heatmap`**: Cambiar la lógica para que no recorra todo el grid en cada frame.
2. **Unificar Configuración**: Auditar `utils/config.py` y `utils/config_manager.py` y fusionar.
3. **Toggle de "Modo Rendimiento"**: Crear una configuración simple en `config.json` llamada `performance_mode` que, si es `true`, desactive automáticamente:
   - Mapas de calor.
   - Partículas.
   - Sombras.
   - Bordes redondeados.
   - Fuentes suavizadas (anti-aliasing).