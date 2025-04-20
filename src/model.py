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
        file_name,
        folder_path="./model_Model",
        n_games=0,
        optimizer=None,
        loss=None,
        last_record_game=None,
        record=None,
        pathfinding_enabled=True,
        temperature=None,
    ):
        from utils.logger import Logger

        os.makedirs(folder_path, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "n_games": n_games,
            "optimizer_state_dict": optimizer,
            "loss": loss,
            "last_record_game": last_record_game,
            "record": record,
            "pathfinding_enabled": pathfinding_enabled,
            "temperature": temperature,
        }
        checkpoint_path = os.path.join(folder_path, file_name)
        try:
            torch.save(checkpoint, checkpoint_path)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            Logger.print_model_saved(checkpoint_path, current_time)
        except Exception as e:
            Logger.print_error(f"Error al guardar el modelo: {e}")

    def load(self, file_name, folder_path="./model_Model"):
        from utils.logger import Logger

        folder_fullpath = folder_path
        if not os.path.exists(folder_fullpath):
            Logger.print_error(f"Directorio {folder_fullpath} no existe.")
            return None, None, None, None, None, None, None

        file_path = os.path.join(folder_fullpath, file_name)
        try:
            checkpoint = torch.load(file_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            n_games = checkpoint.get("n_games", 0)
            loss = checkpoint.get("loss", None)
            optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
            last_record_game = checkpoint.get("last_record_game", 0)
            record = checkpoint.get("record", 0)
            pathfinding_enabled = checkpoint.get("pathfinding_enabled", True)
            temperature = checkpoint.get("temperature", None)

            # Imprimir información del modelo cargado
            Logger.print_section("Modelo Cargado")
            Logger.print_info(f"Checkpoint: '{file_name}'")
            Logger.print_info(f"Cargado desde: {folder_fullpath}")
            Logger.print_metrics_group("Estadísticas del modelo", {
                "Juegos totales": n_games,
                "Récord": record,
                "Pathfinding": 'activado' if pathfinding_enabled else 'desactivado',
                "Temperatura": temperature
            })

            return (
                n_games,
                loss,
                optimizer_state_dict,
                last_record_game,
                record,
                pathfinding_enabled,
                temperature,
            )
        except FileNotFoundError:
            Logger.print_error(f"Archivo {file_path} no encontrado.")
            return None, None, None, None, None, None, None
        except Exception as e:
            Logger.print_error(f"Error al cargar el checkpoint '{file_name}': {e}")
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
