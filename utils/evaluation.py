"""
Módulo de evaluación para el agente Snake DQN.
Proporciona funciones para evaluar periódicamente el rendimiento del agente
en escenarios controlados.
"""

import numpy as np
import torch
import random
from src.game import SnakeGameAI, Direction, Point
from utils.config import MIN_TEMPERATURE

def evaluate_agent(agent, num_episodes=10, seed=42):
    """
    Evalúa el rendimiento del agente en un conjunto fijo de escenarios.

    Args:
        agent: El agente a evaluar
        num_episodes: Número de episodios de evaluación
        seed: Semilla para reproducibilidad

    Returns:
        dict: Diccionario con métricas de evaluación
    """
    # Guardar estado original
    original_temp = agent.temperature
    original_game = agent.game
    original_pathfinding = agent.pathfinding_enabled if hasattr(agent, 'pathfinding_enabled') else True

    # Configurar para evaluación
    agent.temperature = MIN_TEMPERATURE  # Minimizar exploración durante evaluación
    agent.pathfinding_enabled = True  # Usar pathfinding durante evaluación

    # Métricas a recopilar
    evaluation_scores = []
    evaluation_lengths = []
    evaluation_steps = []
    evaluation_rewards = []

    # Crear juego de evaluación con semilla fija
    np.random.seed(seed)
    torch.manual_seed(seed)

    for episode in range(num_episodes):
        # Crear nuevo juego para cada episodio
        # Establecer semilla para reproducibilidad pero sin pasarla como parámetro
        random.seed(seed + episode)  # Variar semilla ligeramente
        game = SnakeGameAI()
        agent.game = game  # Actualizar referencia del juego

        done = False
        episode_rewards = []

        while not done:
            state = agent.get_state(game)
            action = agent.get_action(game, state)
            reward, done, score = game.play_step(action, agent.n_games, agent.record)
            episode_rewards.append(reward)

        # Recopilar métricas
        evaluation_scores.append(score)
        evaluation_lengths.append(len(game.snake))
        evaluation_steps.append(game.steps)
        evaluation_rewards.append(sum(episode_rewards))

    # Restaurar estado original
    agent.temperature = original_temp
    agent.game = original_game
    agent.pathfinding_enabled = original_pathfinding

    # Importar funciones seguras
    from utils.numpy_utils import safe_mean, safe_std

    # Calcular métricas de forma segura
    results = {
        'mean_score': safe_mean(evaluation_scores),
        'max_score': max(evaluation_scores) if evaluation_scores else 0,
        'min_score': min(evaluation_scores) if evaluation_scores else 0,
        'mean_length': safe_mean(evaluation_lengths),
        'mean_steps': safe_mean(evaluation_steps),
        'mean_reward': safe_mean(evaluation_rewards),
        'std_score': safe_std(evaluation_scores),
        'episodes': num_episodes
    }

    return results

def print_evaluation_results(results, is_periodic=True):
    """Imprime los resultados de la evaluación de manera formateada"""
    from utils.logger import Logger

    # Imprimir encabezado de evaluación
    Logger.print_evaluation_header(is_periodic)

    # Imprimir resultados
    Logger.print_evaluation_results(results)
