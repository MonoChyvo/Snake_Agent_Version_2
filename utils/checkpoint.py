import os
import torch
from utils.config import MODEL_DIR, MODEL_FILENAME
from utils.error_logger import agent_logger, log_error, log_info


class CheckpointManager:
    def __init__(self, model, optimizer, folder_prefix=MODEL_DIR):
        self.model = model
        self.optimizer = optimizer
        self.folder_prefix = folder_prefix

        # El nombre del archivo ahora está fuera de la carpeta 'model_Model' por compatibilidad heredada
        # Pero podemos centralizar la ruta aquí
        if not os.path.exists(folder_prefix):
            os.makedirs(folder_prefix)

    def save(
        self,
        n_games,
        last_record_game,
        record,
        pathfinding_enabled,
        temperature,
        filename=MODEL_FILENAME,
    ):
        """Guarda un punto de control del modelo y metadatos del agente."""
        checkpoint = {
            "n_games": n_games,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "last_record_game": last_record_game,
            "record": record,
            "pathfinding_enabled": pathfinding_enabled,
            "temperature": temperature,
        }

        filepath = os.path.join(self.folder_prefix, filename)
        torch.save(checkpoint, filepath)
        # print(f"Checkpoint guardado en {filepath}")

    def load(self, filename=MODEL_FILENAME):
        """Carga un punto de control del modelo."""
        filepath = os.path.join(self.folder_prefix, filename)
        if not os.path.exists(filepath):
            return None

        try:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint.get("model_state_dict"))
            self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))

            return {
                "n_games": checkpoint.get("n_games"),
                "last_record_game": checkpoint.get("last_record_game"),
                "record": checkpoint.get("record"),
                "pathfinding_enabled": checkpoint.get("pathfinding_enabled"),
                "temperature": checkpoint.get("temperature"),
            }
        except Exception as e:
            log_error(
                agent_logger,
                "Checkpoint",
                f"Error cargando checkpoint {filepath}",
                exception=e,
            )
            return None
