#!/bin/bash
# setup_env.sh - Configura el entorno virtual para Snake DQN

echo "Verificando la versión de Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 no está instalado"
    exit 1
fi

echo "Creando entorno virtual..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: No se pudo crear el entorno virtual"
    exit 1
fi

echo "Activando entorno virtual..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: No se pudo activar el entorno virtual"
    exit 1
fi

echo "Actualizando pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Advertencia: No se pudo actualizar pip"
fi

echo "Instalando dependencias..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: No se pudieron instalar las dependencias"
    exit 1
fi

echo ""
echo "Entorno configurado correctamente."
echo "Para activar el entorno en el futuro, ejecute: source venv/bin/activate"
echo "Para verificar el entorno, ejecute: python check_env.py"
echo ""

exit 0
