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
from helper import (
    log_game_results,
    save_checkpoint,
    update_plots,
    print_weight_norms,
    print_game_info,
)
from config import (
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
)
from model import DQN, QTrainer
from colorama import Fore, Style
from game import SnakeGameAI, Direction, Point

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0
        self.memory_stats = {"max_size_mb": 0, "prune_count": 0, "last_size": 0}

    def push(self, experience):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = experience
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == 0:
            raise ValueError("Memory is empty")
        if len(self.priorities) != len(self.memory):
            self.priorities = np.ones(len(self.memory), dtype=np.float32).tolist()

        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        mini_sample = [self.memory[i] for i in indices]
        weights = probabilities[indices]
        return mini_sample, indices, weights

    def update_priorities(self, batch_indices, batch_priorities, max_priority=100.0):
        for idx, priority in zip(batch_indices, batch_priorities):
            clamped_priority = np.clip(priority, a_min=0, a_max=max_priority)
            self.priorities[idx] = clamped_priority

    def get_memory_usage(self):
        """Calculate approximate memory usage of the replay buffer in MB"""
        if not self.memory:
            return 0

        # Sample a few experiences to get average size
        sample_size = min(10, len(self.memory))
        sample_experiences = self.memory[:sample_size]

        # Calculate size in bytes
        total_bytes = sum(sys.getsizeof(exp) for exp in sample_experiences)
        avg_bytes_per_exp = total_bytes / sample_size

        # Estimate total memory usage
        estimated_total_bytes = avg_bytes_per_exp * len(self.memory)
        estimated_mb = estimated_total_bytes / (1024 * 1024)

        # Update stats
        self.memory_stats["last_size"] = int(estimated_mb)
        self.memory_stats["max_size_mb"] = int(
            max(self.memory_stats["max_size_mb"], estimated_mb)
        )

        return estimated_mb

    def prune_memory(self):
        """Prune memory by keeping only the most important experiences"""
        if len(self.memory) < 1000:  # Don't prune if too small
            return

        # Calculate how many experiences to keep
        keep_count = int(len(self.memory) * MEMORY_PRUNE_FACTOR)

        # Sort indices by priority (highest first)
        sorted_indices = np.argsort(self.priorities)[::-1]
        keep_indices = sorted_indices[:keep_count]

        # Keep only the selected experiences and their priorities
        self.memory = [self.memory[i] for i in keep_indices]
        self.priorities = [self.priorities[i] for i in keep_indices]
        self.position = 0  # Reset position

        # Update stats
        self.memory_stats["prune_count"] += 1

        # Force garbage collection
        gc.collect()

        print(
            Fore.YELLOW
            + f"Memory pruned: kept {keep_count} experiences with highest priorities"
            + Style.RESET_ALL
        )


class Agent:
    def __init__(self):
        self.n_games = 0
        self.last_record_game = 0
        self.game = None
        self.record = 0
        self.memory = PrioritizedReplayMemory(MAX_MEMORY)
        self.model = DQN(19, 256, 3).to(device)
        self.target_model = DQN(19, 256, 3).to(device)
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

        try:
            (
                n_games_loaded,
                _,
                optimizer_state_dict,
                last_recorded_game,
                record,
                pathfinding_enabled,
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
        except Exception as e:
            print(f"No previous model loaded or error loading model: {e}")
            # Keep default values initialized above

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

        priorities = np.full(len(indices), loss + 1e-5, dtype=np.float32)
        self.memory.update_priorities(indices, priorities)

        return loss  # Return the loss value

    def train_short_memory(self, state, action, reward, next_state, done):
        action_idx = np.array([np.argmax(action)])
        weights = np.ones((1,), dtype=np.float32)
        self.trainer.train_step(state, action_idx, reward, next_state, done, weights)

    def get_action(self, game, state):
        # Check if the game reference is set and pathfinding is enabled
        if (
            hasattr(self, "game")
            and hasattr(self, "pathfinding_enabled")
            and self.pathfinding_enabled
        ):
            path = game.find_path()
            if path:
                safe_path = game._safe_moves(path)
                if safe_path:
                    next_step = safe_path[1] if len(safe_path) > 1 else safe_path[0]
                    current_head = game._grid_position(game.head)
                    dx = next_step[0] - current_head[0]
                    dy = next_step[1] - current_head[1]

                    # Convert grid direction to game direction
                    target_direction = None
                    if dx == 1:  # Move right
                        target_direction = Direction.RIGHT
                    elif dx == -1:  # Move left
                        target_direction = Direction.LEFT
                    elif dy == 1:  # Move down
                        target_direction = Direction.DOWN
                    elif dy == -1:  # Move up
                        target_direction = Direction.UP

                    # Convert target direction to action based on current direction
                    if target_direction:
                        directions = [
                            Direction.RIGHT,
                            Direction.DOWN,
                            Direction.LEFT,
                            Direction.UP,
                        ]
                        current_idx = directions.index(game.direction)
                        target_idx = directions.index(target_direction)

                        # Calculate the turn needed (0 = straight, 1 = right, -1 or 3 = left)
                        turn = (target_idx - current_idx) % 4

                        if turn == 0:  # No turn needed (straight)
                            return [1, 0, 0]
                        elif turn == 1:  # Turn right
                            return [0, 1, 0]
                        else:  # Turn left (turn == 3 or turn == -1)
                            return [0, 0, 1]

        # Fallback to DQN prediction
        state_tensor = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state_tensor).detach().numpy()

        # Improved numerical stability: subtract max value
        exp_q = np.exp((q_values - np.max(q_values)) / self.temperature)

        probabilities = exp_q / np.sum(exp_q)
        action = np.random.choice(len(q_values), p=probabilities)
        final_move = [0, 0, 0]
        final_move[action] = 1
        return final_move

    def update_target_network(self):
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

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
            self.temperature = max(self.temperature * decay_rate, min_temperature)

    def set_pathfinding(self, enabled=True):
        """Activa o desactiva el pathfinding para la selección de acciones"""
        self.pathfinding_enabled = enabled
        print(
            Fore.CYAN
            + f"Pathfinding {'activado' if enabled else 'desactivado'}"
            + Style.RESET_ALL
        )


def train(max_games: int) -> None:
    agent = Agent()
    # Hacer que el agente sea accesible globalmente
    globals()["agent"] = agent
    game = SnakeGameAI()  # Set the game reference

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

            # Terminate training if max_games reached
            if agent.n_games >= max_games:
                print(Fore.GREEN + "            Training complete." + Style.RESET_ALL)
                break


if __name__ == "__main__":
    train(MAX_EPOCHS)
