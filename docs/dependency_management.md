# Gestión de Dependencias

Esta rama se enfoca en mejorar la gestión de dependencias del proyecto, asegurando la reproducibilidad del entorno y facilitando la instalación para nuevos usuarios.

## Objetivos

1. **Actualizar requirements.txt**

   - Especificar versiones exactas de todas las dependencias
   - Organizar dependencias por categorías
   - Documentar el propósito de cada dependencia

2. **Implementar Gestión de Entorno Virtual**
   - Crear scripts para configurar entornos virtuales
   - Documentar el proceso de configuración
   - Proporcionar soporte para diferentes sistemas operativos

## Tareas Específicas

### Actualización de requirements.txt

- [x] Auditar todas las dependencias actuales
- [x] Determinar las versiones mínimas y máximas compatibles
- [x] Actualizar `requirements.txt` con versiones específicas
- [x] Organizar dependencias en categorías:
  - Dependencias principales
  - Dependencias de visualización
  - Dependencias de desarrollo
  - Dependencias opcionales
- [x] Añadir comentarios explicativos para cada dependencia
- [x] Unificar todas las dependencias en un solo archivo `requirements.txt` con secciones claras y comentarios explicativos

### Gestión de Entorno Virtual

- [x] Crear scripts para configurar entornos virtuales:
  - `setup_env.bat` para Windows
  - `setup_env.sh` para Linux/macOS
- [x] Implementar verificación de versión de Python
- [x] Añadir instrucciones detalladas en el README
- [x] Crear un script de verificación de entorno (`check_env.py`)
- [x] Implementar detección y solución de problemas comunes

## Plan de Implementación

1. **Fase 1: Auditoría de Dependencias**

   - Identificar todas las dependencias utilizadas en el código
   - Determinar qué dependencias son esenciales vs. opcionales
   - Verificar compatibilidad entre versiones

2. **Fase 2: Actualización de requirements.txt**

   - Actualizar el archivo con versiones específicas
   - Organizar y comentar el archivo
   - Crear archivos de requisitos adicionales

3. **Fase 3: Scripts de Entorno**

   - Desarrollar scripts para diferentes sistemas operativos
   - Probar en diferentes entornos
   - Documentar el proceso

4. **Fase 4: Documentación**
   - Actualizar el README con instrucciones detalladas
   - Crear una guía de solución de problemas
   - Documentar el proceso de actualización de dependencias

## Ejemplo de requirements.txt Mejorado

```
# Dependencias principales
pygame==2.1.2        # Motor de juego
torch==1.10.0        # Framework de aprendizaje profundo
numpy==1.21.4        # Procesamiento numérico
pandas==1.3.4        # Análisis de datos

# Visualización
matplotlib==3.5.1    # Generación de gráficos
seaborn==0.11.2      # Visualización estadística avanzada

# Utilidades
tqdm==4.62.3         # Barras de progreso
colorama==0.4.4      # Salida de consola coloreada

# Desarrollo (opcional)
pytest==6.2.5        # Framework de pruebas
flake8==4.0.1        # Linter de código
black==21.12b0       # Formateador de código
```

## Ejemplo de Script de Configuración

```bash
#!/bin/bash
# setup_env.sh - Configura el entorno virtual para Snake DQN

# Verificar versión de Python
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 no está instalado"
    exit 1
fi

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

echo "Entorno configurado correctamente"
```

## Métricas de Éxito

- Instalación exitosa en diferentes sistemas operativos
- Reducción de problemas relacionados con dependencias
- Documentación clara y completa del proceso de instalación
- Capacidad para reproducir exactamente el mismo entorno

## Referencias

- [Guía de pip](https://pip.pypa.io/en/stable/user_guide/)
- [Entornos virtuales de Python](https://docs.python.org/3/tutorial/venv.html)
- [Mejores prácticas para requirements.txt](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
