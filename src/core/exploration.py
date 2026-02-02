import time
from utils.logger import Logger
from src.core.shared_state import update_model_params


class ExplorationStrategy:
    def __init__(
        self,
        initial_temp,
        min_temp,
        decay_rate,
        exploration_freq,
        exploration_temp,
        exploration_duration,
    ):
        self.temperature = initial_temp
        self.min_temp = min_temp
        self.decay_rate = decay_rate
        self.exploration_freq = exploration_freq
        self.exploration_temp = exploration_temp
        self.exploration_duration = exploration_duration

        self.pre_exploration_temp = initial_temp
        self.exploration_phase = False
        self.exploration_games_left = 0
        self.last_exploration_game = 0

    def update(self, current_game, pathfinding_enabled=True):
        """
        Actualiza el estado de exploración basado en el progreso del juego.
        Retorna (temperature, exploration_phase, games_left)
        """
        # Inicia fase de exploración si corresponde
        if (
            current_game > 0
            and current_game % self.exploration_freq == 0
            and current_game > self.last_exploration_game + self.exploration_freq / 2
        ):
            Logger.print_section("Fase de Exploración")
            Logger.print_warning(
                f"!Iniciando fase de exploración por {self.exploration_duration} juegos!"
            )
            self.exploration_phase = True
            self.exploration_games_left = self.exploration_duration
            self.last_exploration_game = current_game
            self.pre_exploration_temp = self.temperature
            self.temperature = self.exploration_temp

            self._sync_shared_state(pathfinding_enabled)
            return self.temperature, self.exploration_phase, self.exploration_games_left

        # Si estamos en fase de exploración, reducir el contador
        if self.exploration_phase:
            self.exploration_games_left -= 1

            if self.exploration_games_left <= 0:
                self.exploration_phase = False
                self.temperature = self.pre_exploration_temp
                Logger.print_warning(
                    f"Fase de exploración terminada. Volviendo a temperatura {self.temperature:.4f}"
                )
                self._sync_shared_state(pathfinding_enabled)
            else:
                # Actualizar solo juegos restantes en el estado compartido
                update_model_params(
                    {
                        "exploration_games_left": self.exploration_games_left,
                        "mode": f"Exploración (restantes: {self.exploration_games_left})",
                    }
                )

            return self.temperature, self.exploration_phase, self.exploration_games_left

        # Ajuste de temperatura normal (decaimiento)
        old_temp = self.temperature
        self.temperature = max(self.temperature * self.decay_rate, self.min_temp)

        # Sincronizar si hubo cambio significativo
        if abs(old_temp - self.temperature) > 0.001:
            Logger.print_info(
                f"Temperatura actualizada: {old_temp:.4f} -> {self.temperature:.4f}"
            )
            update_model_params({"temperature": self.temperature})

        return self.temperature, self.exploration_phase, self.exploration_games_left

    def _sync_shared_state(self, pathfinding_enabled):
        mode = (
            f"Exploración (restantes: {self.exploration_duration})"
            if self.exploration_phase
            else (
                "Pathfinding habilitado"
                if pathfinding_enabled
                else "Explotación normal"
            )
        )

        update_model_params(
            {
                "exploration_phase": self.exploration_phase,
                "exploration_games_left": (
                    self.exploration_games_left if self.exploration_phase else 0
                ),
                "temperature": self.temperature,
                "mode": mode,
            }
        )
