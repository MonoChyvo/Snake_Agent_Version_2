"""
Modelo de red neuronal y componentes de entrenamiento para la implementación del Snake DQN.
Este módulo contiene:

Componentes principales:
- DQN: Arquitectura de Red Q Profunda con funcionalidad de guardado/carga
- QTrainer: Implementación del entrenamiento con:
  * Algoritmo DQN Doble
  * Regularización L2 con coeficientes específicos por capa
  * Seguimiento de gradientes y monitoreo de cambios en pesos
  * Cálculo avanzado de pérdida con muestreo por importancia
  * Gestión de puntos de control para persistencia del modelo

Características:
- Selección automática de dispositivo (CPU/CUDA)
- Sistema de registro completo
- Manejo de errores para operaciones del modelo
- Monitoreo de rendimiento y herramientas de depuración
"""

import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from colorama import Fore, Style
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

# Importar funciones de validación
try:
    from utils.validation import validate_model_file, validate_model_content, safe_model_load
    validation_available = True
except ImportError:
    validation_available = False
    logging.warning("Módulo de validación no disponible. Se omitirá la validación de modelos.")

# Configure logging if it has not been set up
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(
        self,
        file_name: str,
        folder_path: str = "./model_Model",
        n_games: int = 0,
        optimizer: Any = None,
        loss: Optional[float] = None,
        last_record_game: Optional[int] = None,
        record: Optional[int] = None,
        pathfinding_enabled: bool = True,
        temperature: Optional[float] = None,
    ) -> bool:
        """
        Guarda el modelo en un archivo con validación y manejo de excepciones mejorado.

        Args:
            file_name: Nombre del archivo para guardar el modelo
            folder_path: Ruta de la carpeta donde guardar el modelo
            n_games: Número de juegos completados
            optimizer: Estado del optimizador
            loss: Valor de pérdida actual
            last_record_game: Último juego donde se estableció un récord
            record: Puntuación récord actual
            pathfinding_enabled: Si el pathfinding está habilitado
            temperature: Valor de temperatura actual

        Returns:
            bool: True si el modelo se guardó correctamente, False en caso contrario
        """
        try:
            # Validar parámetros de entrada
            if not isinstance(file_name, str) or not file_name:
                raise ValueError("El nombre del archivo no puede estar vacío")

            if not isinstance(folder_path, str):
                raise ValueError("La ruta de la carpeta debe ser una cadena de texto")

            if not file_name.endswith('.pth'):
                file_name += '.pth'  # Asegurar que tenga la extensión correcta

            # Crear directorio si no existe
            os.makedirs(folder_path, exist_ok=True)

            # Preparar el checkpoint con validación de tipos
            checkpoint = {
                "model_state_dict": self.state_dict(),
                "n_games": int(n_games) if n_games is not None else 0,
                "optimizer_state_dict": optimizer,
                "loss": float(loss) if loss is not None else None,
                "last_record_game": int(last_record_game) if last_record_game is not None else 0,
                "record": int(record) if record is not None else 0,
                "pathfinding_enabled": bool(pathfinding_enabled),
                "temperature": float(temperature) if temperature is not None else None,
            }

            # Validar el contenido del checkpoint si está disponible la validación
            if validation_available:
                try:
                    validate_model_content(checkpoint)
                except Exception as e:
                    logging.error(f"Error de validación del checkpoint: {e}")
                    raise

            # Guardar el modelo
            checkpoint_path = os.path.join(folder_path, file_name)
            torch.save(checkpoint, checkpoint_path)

            # Verificar que el archivo se haya guardado correctamente
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"No se pudo guardar el archivo en {checkpoint_path}")

            # Verificar que el archivo no esté vacío
            if os.path.getsize(checkpoint_path) == 0:
                raise ValueError(f"El archivo guardado está vacío: {checkpoint_path}")

            # Registrar éxito
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            success_msg = f"Modelo guardado en {checkpoint_path} el {current_time}"
            logging.info(success_msg)
            print(Fore.CYAN + success_msg + Style.RESET_ALL)
            print(Fore.RED + "-" * 60 + Style.RESET_ALL)
            print("")
            return True

        except ValueError as e:
            error_msg = f"Error de validación al guardar el modelo: {e}"
            logging.error(error_msg)
            print(Fore.RED + error_msg + Style.RESET_ALL)
            return False
        except FileNotFoundError as e:
            error_msg = f"Error de archivo al guardar el modelo: {e}"
            logging.error(error_msg)
            print(Fore.RED + error_msg + Style.RESET_ALL)
            return False
        except Exception as e:
            error_msg = f"Error inesperado al guardar el modelo: {e}"
            logging.error(error_msg)
            print(Fore.RED + error_msg + Style.RESET_ALL)
            return False

    def load(self, file_name: str, folder_path: str = "./model_Model") -> Tuple[Optional[int], Optional[float], Any, Optional[int], Optional[int], Optional[bool], Optional[float]]:
        """
        Carga un modelo desde un archivo con validación y manejo de excepciones mejorado.

        Args:
            file_name: Nombre del archivo del modelo a cargar
            folder_path: Ruta de la carpeta donde se encuentra el modelo

        Returns:
            Tuple: (n_games, loss, optimizer_state_dict, last_record_game, record, pathfinding_enabled, temperature)
                   o (None, None, None, None, None, None, None) si ocurre un error
        """
        try:
            # Validar parámetros de entrada
            if not isinstance(file_name, str) or not file_name:
                raise ValueError("El nombre del archivo no puede estar vacío")

            if not isinstance(folder_path, str):
                raise ValueError("La ruta de la carpeta debe ser una cadena de texto")

            # Asegurar que el archivo tenga la extensión correcta
            if not file_name.endswith('.pth'):
                file_name += '.pth'

            # Verificar que el directorio existe
            folder_fullpath = folder_path
            if not os.path.exists(folder_fullpath):
                error_msg = f"El directorio {folder_fullpath} no existe"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Construir la ruta completa del archivo
            file_path = os.path.join(folder_fullpath, file_name)

            # Validar el archivo si está disponible la validación
            if validation_available:
                try:
                    validate_model_file(file_path)
                    checkpoint = safe_model_load(file_path)
                except Exception as e:
                    logging.error(f"Error de validación del archivo de modelo: {e}")
                    raise
            else:
                # Si no está disponible la validación, cargar directamente
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"El archivo {file_path} no existe")
                checkpoint = torch.load(file_path)

            # Cargar el estado del modelo
            if "model_state_dict" not in checkpoint:
                raise ValueError(f"El archivo {file_path} no contiene un estado de modelo válido")

            self.load_state_dict(checkpoint["model_state_dict"])

            # Extraer valores con validación de tipos
            n_games = int(checkpoint.get("n_games", 0))
            loss = float(checkpoint.get("loss", 0)) if checkpoint.get("loss") is not None else None
            optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
            last_record_game = int(checkpoint.get("last_record_game", 0))
            record = int(checkpoint.get("record", 0))
            pathfinding_enabled = bool(checkpoint.get("pathfinding_enabled", True))
            temperature = float(checkpoint.get("temperature", 0)) if checkpoint.get("temperature") is not None else None

            # Registrar éxito
            success_msg = f"\nUnified checkpoint: '{file_name}' \nLoaded from: {folder_fullpath}. \nTotal games: {n_games}, \nRecord: {record}, \nPathfinding: {'enabled' if pathfinding_enabled else 'disabled'}"
            logging.info(f"Modelo cargado correctamente: {file_path}")
            print(Fore.LIGHTYELLOW_EX + success_msg + Style.RESET_ALL)

            return (
                n_games,
                loss,
                optimizer_state_dict,
                last_record_game,
                record,
                pathfinding_enabled,
                temperature,
            )

        except FileNotFoundError as e:
            error_msg = f"Archivo no encontrado: {e}"
            logging.error(error_msg)
            print(Fore.RED + error_msg + Style.RESET_ALL)
            return None, None, None, None, None, None, None

        except ValueError as e:
            error_msg = f"Error de validación al cargar el modelo: {e}"
            logging.error(error_msg)
            print(Fore.RED + error_msg + Style.RESET_ALL)
            return None, None, None, None, None, None, None

        except torch.serialization.pickle.UnpicklingError as e:
            error_msg = f"El archivo no es un modelo PyTorch válido: {e}"
            logging.error(error_msg)
            print(Fore.RED + error_msg + Style.RESET_ALL)
            return None, None, None, None, None, None, None

        except Exception as e:
            error_msg = f"Error inesperado al cargar el modelo '{file_name}': {e}"
            logging.error(error_msg)
            print(Fore.RED + error_msg + Style.RESET_ALL)
            return None, None, None, None, None, None, None


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # Añadir coeficientes de regularización específicos por capa
        self.l2_lambdas = {"linear1": 0.0001, "linear2": 0.001, "linear3": 0.0001}

    def _prepare_tensor(self, data, dtype, unsqueeze=True):
        tensor = torch.tensor(data, dtype=dtype).to(device)
        if unsqueeze and len(tensor.shape) == 1:
            tensor = torch.unsqueeze(tensor, 0)
        return tensor

    def train_step(self, state, action, reward, next_state, done, weights):
        # Convierte las entradas utilizando la función auxiliar
        state = self._prepare_tensor(state, torch.float32)
        next_state = self._prepare_tensor(next_state, torch.float32)
        action = self._prepare_tensor(action, torch.int64)
        reward = self._prepare_tensor(reward, torch.float32)
        done = self._prepare_tensor(done, torch.float32)

        # Asegurarse de que weights sea un tensor 1D
        if isinstance(weights, np.ndarray) and weights.ndim > 1:
            weights = weights.flatten()
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Asegura que action tenga forma [batch, 1]
        if action.dim() == 1:
            action = action.unsqueeze(1)

        # Calcula predicciones y objetivos
        pred = self.model(state).gather(1, action).squeeze(-1)
        next_action = self.model(next_state).argmax(1).unsqueeze(-1)
        next_pred = self.target_model(next_state).gather(1, next_action).squeeze(-1)
        target = reward + (1 - done) * self.gamma * next_pred

        # Añadir regularización L2 con diferentes intensidades por capa
        l2_reg = 0
        for name, param in self.model.named_parameters():
            if "weight" in name:
                layer_name = name.split(".")[
                    0
                ]  # Obtiene 'linear1', 'linear2' o 'linear3'
                if layer_name in self.l2_lambdas:
                    l2_reg += self.l2_lambdas[layer_name] * param.pow(2).sum()

        # Calcula pérdida y actualiza pesos. Se utiliza target.detach() para evitar la retropropagación a la red target.
        loss = (weights * (pred - target.detach()).pow(2)).mean() + l2_reg

        # Obtén estadísticas previas (p.ej. pesos antes de actualizar)
        old_weights = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

        self.optimizer.zero_grad()
        loss.backward()

        # Logear la magnitud de los gradientes
        grad_norms = {
            name: param.grad.norm().item()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }
        logging.debug(f"Gradient norms: {grad_norms}")

        self.optimizer.step()

        # Calcular y loguear la contribución de L2 a la pérdida
        l2_contributions = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:
                layer_name = name.split(".")[0]
                if layer_name in self.l2_lambdas:
                    l2_contributions[layer_name] = (
                        self.l2_lambdas[layer_name] * param.pow(2).sum().item()
                    )

        # Calcula y logea los cambios en los pesos
        weight_changes = {
            name: (param - old_weights[name]).norm().item()
            for name, param in self.model.named_parameters()
        }
        logging.debug(f"Weight changes: {weight_changes}")

        logging.debug(f"Train step completed. Loss: {loss.item()}")
        return loss.item()
