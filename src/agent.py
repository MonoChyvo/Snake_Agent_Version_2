"""
Implementación principal del agente para el proyecto Snake DQN (Red Q Profunda).
Este módulo contiene:
- PrioritizedReplayMemory: Implementación de repetición de experiencia priorizada
- Agent: Agente DQN principal con exploración basada en temperatura
- Implementación del bucle de entrenamiento

Características principales:
- DQN Doble con red objetivo
- Repetición de experiencia priorizada para mejor eficiencia de muestreo
- Estrategia de exploración dinámica basada en temperatura
- Fases de exploración periódicas para mejor descubrimiento de políticas
- Seguimiento completo de métricas y puntos de control
- Representación de estado con detección de peligro y ubicación de comida
"""

import torch
import psutil
import sys
import gc
import numpy as np
import pygame
from utils.helper import (
    log_game_results,
    save_checkpoint,
    update_plots,
    print_weight_norms,
    print_game_info,
)
from utils.evaluation import evaluate_agent, print_evaluation_results
from utils.efficient_memory import EfficientPrioritizedReplayMemory
from utils.config import (
    MAX_MEMORY,
    LR,
    GAMMA,
    TEMPERATURE,
    EXPLORATION_PHASE,
    MIN_TEMPERATURE,
    DECAY_RATE,
    EXPLORATION_FREQUENCY,
    EXPLORATION_TEMP,
    EXPLORATION_DURATION,
    MAX_EPOCHS,
    BLOCK_SIZE,
    BATCH_SIZE,
    TAU,
    MEMORY_PRUNE_FACTOR,
    MEMORY_MONITOR_FREQ,
    MEMORY_PRUNE_THRESHOLD,
    VISUAL_MODE,
    SHOW_GRID,
    SHOW_HEATMAP,
    PARTICLE_EFFECTS,
    SHADOW_EFFECTS
)
from src.model import DQN, QTrainer
from colorama import Fore, Style
from src.game import SnakeGameAI, Direction, Point
from utils.advanced_pathfinding import AdvancedPathfinding
from src.start_screen import get_user_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# La clase PrioritizedReplayMemory ha sido reemplazada por EfficientPrioritizedReplayMemory
# que se importa desde utils.efficient_memory


class Agent:
    def __init__(self):
        self.n_games = 0
        self.last_record_game = 0
        self.game = None
        self.record = 0
        # Usar la implementación eficiente de memoria con dimensiones actualizadas
        self.memory = EfficientPrioritizedReplayMemory(MAX_MEMORY, state_dim=23, action_dim=3)
        # Dimensión de entrada: 23 (19 originales + 4 nuevas características)
        self.model = DQN(23, 256, 3).to(device)
        self.target_model = DQN(23, 256, 3).to(device)
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=GAMMA)

        self.temperature = TEMPERATURE
        self.pre_exploration_temp = TEMPERATURE
        self.exploration_phase = EXPLORATION_PHASE
        self.exploration_games_left = 0
        self.last_exploration_game = 0

        # Inicializar diccionario para seguimiento de las mejores métricas
        self.best_metrics = {}
        # Lista para mantener los scores recientes para cálculos de mejora
        self.recent_scores = []

        # Intentar cargar el modelo si existe, de lo contrario iniciar con un modelo nuevo
        try:
            # Verificar si el archivo del modelo existe
            import os
            model_path = os.path.join("./model_Model", "model_MARK_IX.pth")
            if os.path.exists(model_path):
                (
                    n_games_loaded,
                    _,
                    optimizer_state_dict,
                    last_recorded_game,
                    record,
                    pathfinding_enabled,
                    temperature,
                ) = self.model.load("model_MARK_IX.pth")

                if n_games_loaded is not None:
                    self.n_games = n_games_loaded
                if last_recorded_game is not None:
                    self.last_record_game = last_recorded_game
                if record is not None:
                    self.record = record
                self.pathfinding_enabled = (
                    pathfinding_enabled if pathfinding_enabled is not None else True
                )
                if temperature is not None:
                    self.temperature = temperature
                    self.pre_exploration_temp = temperature
                    print(
                        Fore.LIGHTYELLOW_EX
                        + f"Temperatura restaurada del checkpoint: {self.temperature:.4f}"
                        + Style.RESET_ALL
                    )
                if optimizer_state_dict is not None:
                    try:
                        self.trainer.optimizer.load_state_dict(optimizer_state_dict)
                        print(
                            Fore.LIGHTYELLOW_EX
                            + "Restored optimizer state from checkpoint"
                            + Style.RESET_ALL
                        )
                        print(Fore.RED + "-" * 60 + Style.RESET_ALL)
                    except Exception as e:
                        print(f"Error restoring optimizer state: {e}")
            else:
                print(Fore.YELLOW + "No se encontró un modelo previo. Iniciando entrenamiento desde cero." + Style.RESET_ALL)
                self.pathfinding_enabled = True
        except Exception as e:
            print(f"Error al intentar cargar el modelo: {e}")
            print(Fore.YELLOW + "Iniciando entrenamiento desde cero." + Style.RESET_ALL)
            self.pathfinding_enabled = True

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.game_results = []

    def monitor_memory(self):
        """Monitor memory usage and prune if necessary"""
        if self.n_games % MEMORY_MONITOR_FREQ != 0:
            return

        # Get memory usage of replay buffer
        buffer_size_mb = self.memory.get_memory_usage()

        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_used_percent = system_memory.percent

        print(Fore.CYAN + f"Memory Monitor - Game {self.n_games}:" + Style.RESET_ALL)
        print(
            f"  Replay Buffer: {buffer_size_mb:.2f} MB ({len(self.memory.memory)} experiences)"
        )
        print(f"  System Memory: {system_used_percent:.1f}% used")

        # Check if buffer is getting too large relative to capacity
        buffer_fill_ratio = len(self.memory.memory) / MAX_MEMORY

        if buffer_fill_ratio > MEMORY_PRUNE_THRESHOLD:
            print(
                Fore.YELLOW
                + f"  Buffer fill ratio ({buffer_fill_ratio:.2f}) exceeds threshold, pruning..."
                + Style.RESET_ALL
            )
            self.memory.prune_memory()

        # Log memory stats
        if hasattr(self, "game_results") and len(self.game_results) > 0:
            self.game_results[-1]["memory_usage_mb"] = buffer_size_mb
            self.game_results[-1]["system_memory_percent"] = system_used_percent

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]

        directions = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ]

        for dx, dy in directions:
            body_detected = False
            for dist in range(1, 5):  # Buscar hasta 4 bloques de distancia
                check_x = head.x + dx * dist * BLOCK_SIZE
                check_y = head.y + dy * dist * BLOCK_SIZE
                check_pt = Point(check_x, check_y)
                if check_pt in game.snake[1:]:
                    state.append(1 / dist)  # Más cercano = valor más alto
                    body_detected = True
                    break
            if not body_detected:
                state.append(0)

        # Características adicionales para mejorar la representación del estado

        # 1. Distancia normalizada a la cola
        tail = game.snake[-1]
        tail_distance = abs(head.x - tail.x) + abs(head.y - tail.y)
        max_possible_distance = game.width + game.height
        normalized_tail_distance = tail_distance / max_possible_distance
        state.append(normalized_tail_distance)

        # 2. Densidad de la serpiente (qué tan compacta está)
        snake_length = len(game.snake)
        grid_area = (game.width // BLOCK_SIZE) * (game.height // BLOCK_SIZE)
        density = snake_length / grid_area
        state.append(density)

        # 3. Longitud normalizada de la serpiente
        max_possible_length = grid_area
        normalized_length = snake_length / max_possible_length
        state.append(normalized_length)

        # 4. Espacio libre alrededor de la cabeza (conectividad)
        free_directions = 0
        for check_pt in [point_l, point_r, point_u, point_d]:
            if not game.is_collision(check_pt):
                free_directions += 1
        normalized_free_directions = free_directions / 4.0
        state.append(normalized_free_directions)

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.push(experience)

    def train_long_memory(self):
        mini_sample, indices, weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        actions = np.array([np.argmax(a) for a in actions])
        states = np.array(states)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        weights = np.array(weights)

        # Normaliza las recompensas
        if len(rewards) > 10:  # Solo normaliza si hay suficientes ejemplos
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        loss = self.trainer.train_step(
            states, actions, rewards, next_states, dones, weights
        )

        # Crear prioridades como una lista de escalares para evitar problemas de dimensionalidad
        priorities = [loss + 1e-5 for _ in range(len(indices))]
        self.memory.update_priorities(indices, priorities)

        return loss  # Return the loss value

    def train_short_memory(self, state, action, reward, next_state, done):
        action_idx = np.array([np.argmax(action)])
        weights = np.ones((1,), dtype=np.float32)

        # Convertir reward a array para consistencia con train_long_memory
        reward_array = np.array([reward])

        # Normalizar la recompensa si es posible (usando estadísticas recientes)
        if hasattr(self, 'recent_rewards') and len(self.recent_rewards) > 10:
            reward_mean = np.mean(self.recent_rewards)
            reward_std = np.std(self.recent_rewards) + 1e-8
            reward_array = (reward_array - reward_mean) / reward_std

        # Mantener un historial de recompensas recientes para normalización
        if not hasattr(self, 'recent_rewards'):
            self.recent_rewards = []
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:  # Limitar a las 100 recompensas más recientes
            self.recent_rewards.pop(0)

        self.trainer.train_step(state, action_idx, reward_array[0], next_state, done, weights)

    def get_action(self, game, state):
        # Intenta encontrar un camino óptimo si el pathfinding está habilitado
        if hasattr(self, 'pathfinding_enabled') and self.pathfinding_enabled:
            optimal_path = game.pathfinder.find_optimal_path()
            if optimal_path:
                # Verifica si el camino es seguro
                safe_path = game._safe_moves(optimal_path)
                if safe_path:
                    # Lógica para decidir la acción basada en el camino óptimo
                    next_pos = safe_path[0]
                    current_pos = game._grid_position(game.head)
                    dx = next_pos[0] - current_pos[0]
                    dy = next_pos[1] - current_pos[1]
                    if dx == 1:
                        return [0, 1, 0]  # Derecha
                    elif dx == -1:
                        return [0, 0, 1]  # Izquierda
                    elif dy == 1:
                        return [1, 0, 0]  # Abajo
                    elif dy == -1:
                        return [0, 1, 0]  # Arriba

        # Calcular temperatura adaptativa basada en la incertidumbre del modelo
        adaptive_temp = self.get_adaptive_temperature(state)

        # Fallback a la predicción DQN con temperatura adaptativa
        state_tensor = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state_tensor).detach().numpy()

        # Improved numerical stability: subtract max value
        exp_q = np.exp((q_values - np.max(q_values)) / adaptive_temp)

        probabilities = exp_q / np.sum(exp_q)
        action = np.random.choice(len(q_values), p=probabilities)
        final_move = [0, 0, 0]
        final_move[action] = 1
        return final_move

    def get_adaptive_temperature(self, state):
        """Calcula una temperatura adaptativa basada en la incertidumbre del modelo"""
        # Si estamos en fase de exploración, usar temperatura de exploración
        if self.exploration_phase:
            return self.temperature

        # Calcular incertidumbre basada en la longitud de la serpiente
        # Serpientes más largas necesitan más exploración para evitar quedarse atrapadas
        if hasattr(self, 'game') and self.game is not None:
            snake_length = len(self.game.snake)
            # Aumentar temperatura para serpientes más largas
            length_factor = min(snake_length / 20.0, 1.0)  # Normalizar a máximo 1.0
            base_temp = self.temperature
            # Aumentar temperatura hasta un 50% para serpientes largas
            adaptive_temp = base_temp * (1.0 + 0.5 * length_factor)

            # Limitar el rango de temperatura
            # Asegurarse de que adaptive_temp sea un escalar
            adaptive_temp_scalar = float(adaptive_temp)
            return max(min(adaptive_temp_scalar, 1.0), MIN_TEMPERATURE)

        # Si no podemos calcular basado en longitud, usar temperatura estándar
        return self.temperature

    def update_target_network(self):
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    def adjust_learning_rate(self, initial_lr=LR, min_lr=0.00001):
        """Ajusta la tasa de aprendizaje basada en el progreso del entrenamiento"""
        # Calcular factor de decaimiento basado en el número de juegos
        decay_factor = 0.9999

        # Calcular nueva tasa de aprendizaje
        new_lr = max(initial_lr * (decay_factor ** self.n_games), min_lr)

        # Aplicar nueva tasa de aprendizaje al optimizador
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = new_lr

        return new_lr

    def update_temperature(
        self,
        decay_rate,
        min_temperature,
        current_game,
        exploration_temp,
        exploration_frequency,
        exploration_duration,
    ):
        # Inicia fase de exploración si corresponde
        if (
            current_game > 0
            and current_game % exploration_frequency == 0
            and current_game > self.last_exploration_game + exploration_frequency / 2
        ):
            print(
                Fore.YELLOW
                + f"¡Iniciando fase de exploración por {exploration_duration} juegos!"
                + Style.RESET_ALL
            )
            self.exploration_phase = True
            self.exploration_games_left = exploration_duration
            self.last_exploration_game = current_game
            self.pre_exploration_temp = self.temperature
            self.temperature = exploration_temp
            return

        # Si estamos en fase de exploración, reducir el contador
        if self.exploration_phase:
            self.exploration_games_left -= 1
            if self.exploration_games_left <= 0:
                self.exploration_phase = False
                self.temperature = self.pre_exploration_temp
                print(
                    Fore.YELLOW
                    + f"Fase de exploración terminada. Volviendo a temperatura {self.temperature:.4f}"
                    + Style.RESET_ALL
                )
                return

        # Ajuste de temperatura fuera de fase de exploración
        if not self.exploration_phase:
            old_temp = self.temperature
            self.temperature = max(self.temperature * decay_rate, min_temperature)
            # Mostrar cambio de temperatura solo si es significativo
            if abs(old_temp - self.temperature) > 0.001:
                print(
                    Fore.CYAN
                    + f"Temperatura actualizada: {old_temp:.4f} -> {self.temperature:.4f}"
                    + Style.RESET_ALL
                )

    def set_pathfinding(self, enabled=True):
        """Activa o desactiva el pathfinding para la selección de acciones"""
        self.pathfinding_enabled = enabled
        print(
            Fore.CYAN
            + f"Pathfinding {'activado' if enabled else 'desactivado'}"
            + Style.RESET_ALL
        )


def train(max_games: int) -> None:
    # Inicializar pygame si no está inicializado
    if not pygame.get_init():
        pygame.init()

    # Mostrar pantalla de inicio para seleccionar configuración visual
    visual_config = get_user_config()
    print(Fore.GREEN + "Configuración visual seleccionada:" + Style.RESET_ALL)
    for key, value in visual_config.items():
        print(f"  {key}: {value}")

    agent = Agent()
    # Hacer que el agente sea accesible globalmente
    globals()["agent"] = agent

    # Crear el juego con la configuración visual seleccionada
    game = SnakeGameAI(visual_config=visual_config)  # Set the game reference
    pathfinder = AdvancedPathfinding(game)

    # Establecer la referencia al juego en el agente para cálculos adaptativos
    agent.game = game

    record = agent.record if hasattr(agent, "record") else 0
    total_score = 0
    plot_mean_scores = []
    plot_scores = []

    # Temperature decay settings
    min_temperature = MIN_TEMPERATURE
    decay_rate = DECAY_RATE

    # Exploración periódica (nuevos parámetros)
    exploration_frequency = EXPLORATION_FREQUENCY
    exploration_temp = EXPLORATION_TEMP
    exploration_duration = EXPLORATION_DURATION

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(game, state_old)
        reward, done, score = game.play_step(final_move, agent.n_games, record)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        # Perform soft update of target network every step
        agent.update_target_network()

        if done:
            # Calculate reward statistics for the current game
            episode_reward = sum(game.reward_history)
            avg_reward = (
                episode_reward / len(game.reward_history) if game.reward_history else 0
            )

            # Añadir score actual a la lista de scores recientes para cálculos de mejora
            if not hasattr(agent, "recent_scores"):
                agent.recent_scores = []
            agent.recent_scores.append(score)
            if len(agent.recent_scores) > 50:  # Mantener solo los últimos 50 scores
                agent.recent_scores.pop(0)

            # Long term training and capturing loss
            loss = agent.train_long_memory()

            # Guardar datos del juego actual para análisis
            log_game_results(agent, score, record, game, avg_loss=loss)

            # Actualizar temperatura de exploración
            agent.update_temperature(
                decay_rate,
                min_temperature,
                agent.n_games,
                exploration_temp,
                exploration_frequency,
                exploration_duration,
            )

            # Ajustar tasa de aprendizaje dinámicamente
            current_lr = agent.adjust_learning_rate()
            print(Fore.CYAN + f"Learning rate ajustada a: {current_lr:.6f}" + Style.RESET_ALL)

            # Actualizar contador de juegos y resetear el juego
            agent.n_games += 1
            agent.monitor_memory()
            game.reset()

            # Guardar checkpoint regular si es mejor record
            if score > record:
                record = score
                agent.record = score
                agent.last_record_game = agent.n_games

                # Guardar un checkpoint basado en record
                print(
                    Fore.CYAN
                    + f"New record at game: {agent.last_record_game}!"
                    + Style.RESET_ALL
                )

            save_checkpoint(agent, loss)

            # Actualizar gráficas
            total_score = update_plots(
                agent, score, total_score, plot_scores, plot_mean_scores
            )

            # Auxiliary functions to print information
            print(Fore.RED + "-" * 60 + Style.RESET_ALL)
            print(
                Fore.LIGHTYELLOW_EX
                + f"                    || Game {agent.n_games} ||"
                + Style.RESET_ALL
            )
            print(
                Fore.LIGHTMAGENTA_EX + f"Ended with loss: {loss:.4f}" + Style.RESET_ALL
            )
            print(
                Fore.LIGHTMAGENTA_EX
                + f"Total Reward: {episode_reward:.4f} \nAvg Reward: {avg_reward:.2f}"
                + Style.RESET_ALL
            )
            print(
                Fore.MAGENTA
                + f"Current temperature: {agent.temperature:.4f}"
                + Style.RESET_ALL
            )

            if agent.exploration_phase:
                agent.set_pathfinding(False)
                print(
                    Fore.YELLOW
                    + f"En fase de exploración: {agent.exploration_games_left} juegos restantes"
                    + Style.RESET_ALL
                )
            else:
                agent.set_pathfinding(True)

            print_weight_norms(agent)
            print_game_info(
                episode_reward,
                score,
                agent.last_record_game,
                record,
                agent.recent_scores,
            )

            # Evaluación periódica cada 100 juegos
            if agent.n_games % 100 == 0:
                print(Fore.LIGHTMAGENTA_EX + "\nRealizando evaluación periódica..." + Style.RESET_ALL)
                eval_results = evaluate_agent(agent, num_episodes=5)
                print_evaluation_results(eval_results)

                # Guardar resultados de evaluación
                if hasattr(agent, 'evaluation_results'):
                    agent.evaluation_results.append({
                        'game': agent.n_games,
                        **eval_results
                    })
                else:
                    agent.evaluation_results = [{
                        'game': agent.n_games,
                        **eval_results
                    }]

            # Terminate training if max_games reached
            if agent.n_games >= max_games:
                print(Fore.GREEN + "            Training complete." + Style.RESET_ALL)

                # Evaluación final
                print(Fore.LIGHTMAGENTA_EX + "\nRealizando evaluación final..." + Style.RESET_ALL)
                final_eval = evaluate_agent(agent, num_episodes=10)
                print_evaluation_results(final_eval)
                break


if __name__ == "__main__":
    train(MAX_EPOCHS)
