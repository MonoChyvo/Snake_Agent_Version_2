from typing import Tuple
import numpy as np
from utils.config import (
    BASE_PENALTY,
    FOOD_REWARD,
    LENGTH_BONUS_MULTIPLIER,
    DISTANCE_REWARD_MULTIPLIER,
    SURVIVAL_REWARD,
    DANGER_REWARD,
    FUTURE_PENALTY_MULTIPLIER,
)


class RewardSystem:
    def __init__(self):
        # Hiperparámetros de recompensas cargados desde config
        self.base_penalty = BASE_PENALTY
        self.food_reward = FOOD_REWARD
        self.length_bonus_multiplier = LENGTH_BONUS_MULTIPLIER
        self.distance_reward_multiplier = DISTANCE_REWARD_MULTIPLIER
        self.survival_reward = SURVIVAL_REWARD
        self.danger_reward = DANGER_REWARD
        self.future_penalty_multiplier = FUTURE_PENALTY_MULTIPLIER

    def calculate_reward(self, game_state: dict) -> Tuple[float, bool]:
        """
        Calcula la recompensa basada en el estado actual del juego.
        SIMPLIFICADO: Comer +1, Morir -1, Paso 0.
        """
        reward = 0.0
        game_over = False

        collision = game_state.get("collision", False)
        timeout = game_state.get("timeout", False)

        # Muerte o fin de episodio
        if collision or timeout:
            game_over = True
            reward = -1.0  # Morir -> -1.0
            return reward, game_over

        # Comer comida
        if game_state.get("head_eats_food", False):
            reward = 1.0  # Comer -> +1.0
            return reward, game_over

        # Paso normal (Paso normal -> 0.0)
        reward = 0.0
        return float(reward), game_over
