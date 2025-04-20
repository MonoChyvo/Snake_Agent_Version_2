@echo off
REM setup_env.bat - Configura el entorno virtual para Snake DQN

echo Verificando la versión de Python...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python no está instalado o no está en el PATH
    exit /b 1
)

echo Creando entorno virtual...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Error: No se pudo crear el entorno virtual
    exit /b 1
)

echo Activando entorno virtual...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Error: No se pudo activar el entorno virtual
    exit /b 1
)

echo Actualizando pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Advertencia: No se pudo actualizar pip
)

echo Instalando dependencias...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error: No se pudieron instalar las dependencias
    exit /b 1
)

echo.
echo Entorno configurado correctamente.
echo Para activar el entorno en el futuro, ejecute: venv\Scripts\activate
echo Para verificar el entorno, ejecute: python check_env.py
echo.

exit /b 0
