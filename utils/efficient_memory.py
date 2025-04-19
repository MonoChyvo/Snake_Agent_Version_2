"""
Implementación eficiente de memoria de repetición priorizada utilizando arrays de NumPy.
Esta implementación reduce significativamente el uso de memoria y mejora el rendimiento
de muestreo.
"""

import numpy as np
import gc
import sys
from colorama import Fore, Style
import psutil
from utils.config import MEMORY_PRUNE_FACTOR

class EfficientPrioritizedReplayMemory:
    def __init__(self, capacity, state_dim=23, action_dim=3):
        """
        Inicializa una memoria de repetición priorizada eficiente.

        Args:
            capacity: Capacidad máxima de la memoria
            state_dim: Dimensión del estado (por defecto 23)
            action_dim: Dimensión de la acción (por defecto 3)
        """
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Preasignar arrays para todas las experiencias
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

        # Estadísticas de memoria
        self.memory_stats = {"max_size_mb": 0, "prune_count": 0, "last_size": 0}

    def push(self, experience):
        """
        Añade una experiencia a la memoria.

        Args:
            experience: Tupla (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = experience

        # Obtener el índice donde guardar la experiencia
        idx = self.position

        # Guardar la experiencia en los arrays
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        # Usar la prioridad máxima actual para la nueva experiencia
        max_priority = np.max(self.priorities[:self.size]) if self.size > 0 else 1.0
        self.priorities[idx] = max_priority

        # Actualizar posición y tamaño
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Muestrea un lote de experiencias basado en prioridades.

        Args:
            batch_size: Tamaño del lote a muestrear

        Returns:
            Tupla (mini_sample, indices, weights)
        """
        if self.size == 0:
            raise ValueError("Memory is empty")

        # Calcular probabilidades basadas en prioridades
        probs = self.priorities[:self.size] / np.sum(self.priorities[:self.size])

        # Muestrear índices basados en probabilidades
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Extraer experiencias
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        # Calcular pesos para el muestreo por importancia
        weights = probs[indices]

        # Crear mini-batch
        mini_sample = list(zip(states, actions, rewards, next_states, dones))

        return mini_sample, indices, weights

    def update_priorities(self, batch_indices, batch_priorities, max_priority=100.0):
        """
        Actualiza las prioridades de las experiencias.

        Args:
            batch_indices: Índices de las experiencias a actualizar
            batch_priorities: Nuevas prioridades
            max_priority: Prioridad máxima permitida
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            if idx < self.size:  # Asegurar que el índice es válido
                # Asegurarse de que priority sea un escalar
                if hasattr(priority, 'shape') and priority.shape:  # Si es un array
                    priority_scalar = float(priority.item() if hasattr(priority, 'item') else priority[0])
                else:
                    priority_scalar = float(priority)
                self.priorities[idx] = max(0, min(priority_scalar, max_priority))

    def get_memory_usage(self):
        """
        Calcula el uso aproximado de memoria en MB.

        Returns:
            float: Uso de memoria en MB
        """
        # Calcular tamaño en bytes de los arrays principales
        states_size = self.states.nbytes
        actions_size = self.actions.nbytes
        rewards_size = self.rewards.nbytes
        next_states_size = self.next_states.nbytes
        dones_size = self.dones.nbytes
        priorities_size = self.priorities.nbytes

        # Tamaño total en MB
        total_size_mb = (states_size + actions_size + rewards_size +
                         next_states_size + dones_size + priorities_size) / (1024 * 1024)

        # Actualizar estadísticas
        self.memory_stats["last_size"] = int(total_size_mb)
        self.memory_stats["max_size_mb"] = max(self.memory_stats["max_size_mb"], total_size_mb)

        return total_size_mb

    def prune_memory(self):
        """
        Elimina experiencias con baja prioridad para liberar memoria.
        """
        if self.size < 1000:  # No podar si hay pocas experiencias
            return

        # Calcular cuántas experiencias mantener
        keep_count = int(self.size * MEMORY_PRUNE_FACTOR)

        # Obtener índices ordenados por prioridad (mayor a menor)
        sorted_indices = np.argsort(self.priorities[:self.size])[::-1]
        keep_indices = sorted_indices[:keep_count]

        # Crear arrays temporales para almacenar las experiencias a mantener
        temp_states = self.states[keep_indices].copy()
        temp_actions = self.actions[keep_indices].copy()
        temp_rewards = self.rewards[keep_indices].copy()
        temp_next_states = self.next_states[keep_indices].copy()
        temp_dones = self.dones[keep_indices].copy()
        temp_priorities = self.priorities[keep_indices].copy()

        # Actualizar los arrays principales
        self.states[:keep_count] = temp_states
        self.actions[:keep_count] = temp_actions
        self.rewards[:keep_count] = temp_rewards
        self.next_states[:keep_count] = temp_next_states
        self.dones[:keep_count] = temp_dones
        self.priorities[:keep_count] = temp_priorities

        # Actualizar tamaño y posición
        self.size = keep_count
        self.position = keep_count % self.capacity

        # Actualizar estadísticas
        self.memory_stats["prune_count"] += 1

        # Forzar recolección de basura
        gc.collect()

        print(
            Fore.YELLOW
            + f"Memory pruned: kept {keep_count} experiences with highest priorities"
            + Style.RESET_ALL
        )

    def __len__(self):
        """Retorna el número actual de experiencias en memoria."""
        return self.size
