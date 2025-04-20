import threading
import copy

class StatsManager:
    """
    Clase centralizada para la gestión eficiente de estadísticas del juego.
    - Actualiza y almacena estadísticas relevantes.
    - Notifica cambios solo cuando hay modificaciones reales.
    - Optimiza el rendimiento evitando cálculos y renderizados innecesarios.
    """
    def __init__(self, event_system, game_ref, agent_ref=None):
        self.event_system = event_system
        self.game = game_ref
        self.agent = agent_ref
        self.data = {}
        self._lock = threading.Lock()
        self._dirty = True
        # Suscribirse a eventos relevantes
        self.event_system.register_listener("stats_update_needed", self.update)

    def update(self, event_data=None):
        with self._lock:
            prev_data = self.data.copy() if self.data else None
            # --- Recolectar datos básicos del juego ---
            score = getattr(self.game, 'score', 0)
            record = getattr(self.game, 'record', 0)
            steps = getattr(self.game, 'steps', 0)
            reward_history = getattr(self.game, 'reward_history', [])
            # --- Datos del agente ---
            agent = self.agent
            last_record_game = getattr(agent, 'last_record_game', 0) if agent else 0
            recent_scores = getattr(agent, 'recent_scores', []) if agent else []
            temperature = getattr(agent, 'temperature', 0.0) if agent else 0.0
            pathfinding_enabled = getattr(agent, 'pathfinding_enabled', False) if agent else False
            # Obtener learning rate real desde el optimizador si existe
            if agent and hasattr(agent, 'learning_rate'):
                learning_rate = getattr(agent, 'learning_rate', 0.001)
            elif agent and hasattr(agent, 'trainer') and hasattr(agent.trainer, 'optimizer'):
                learning_rate = 0.001
                for param_group in agent.trainer.optimizer.param_groups:
                    if 'lr' in param_group:
                        learning_rate = param_group['lr']
                        break
            else:
                learning_rate = 0.001
            mode = getattr(agent, 'mode', 'Pathfinding habilitado') if agent else 'Pathfinding habilitado'
            loss = getattr(agent, 'last_loss', 0.0) if agent else 0.0
            # Acciones (si existen)
            straight_pct = getattr(agent, 'straight_pct', 0.0) if agent else 0.0
            right_pct = getattr(agent, 'right_pct', 0.0) if agent else 0.0
            left_pct = getattr(agent, 'left_pct', 0.0) if agent else 0.0
            # --- Métricas adicionales ---
            avg_reward = float(sum(reward_history)/len(reward_history)) if reward_history else 0.0
            # --- Métricas de eficiencia calculadas en tiempo real ---
            # Usar SIEMPRE los valores del agente si existen para que los tests sean estrictos
            efficiency_ratio = getattr(agent, 'efficiency_ratio', 0.0) if agent else 0.0
            # Priorizar steps_per_food del agente si existe, si no calcular
            steps_per_food = getattr(agent, 'steps_per_food', None)
            if steps_per_food is None:
                steps_per_food = steps / score if score > 0 else 0.0
            # --- Construir estructura jerárquica esperada por el panel ---
            new_data = {
                'basic': {
                    'Puntuación': score,
                    'Récord': record,
                    'Pasos': steps,
                },
                'efficiency': {
                    'Ratio de eficiencia': efficiency_ratio,
                    'Pasos por comida': steps_per_food,
                },
                'actions': {
                    'Recto %': getattr(agent, 'straight_pct', 0.0) if agent else 0.0,
                    'Derecha %': getattr(agent, 'right_pct', 0.0) if agent else 0.0,
                    'Izquierda %': getattr(agent, 'left_pct', 0.0) if agent else 0.0,
                },
                'training': {
                    'Recompensa media': avg_reward,
                    'Último récord (juego)': last_record_game,
                },
                'model': {
                    'Pérdida': loss,
                    'Temperatura': temperature,
                    'Learning rate': learning_rate,
                    'Pathfinding': 'Activado' if pathfinding_enabled else 'Desactivado',
                    'Modo de explotación': mode,
                },
            }
            print("[STATS_MANAGER] Datos de estadísticas actuales:", new_data)
            # LOG: datos recolectados para el panel
            print(f"[STATS_MANAGER] Panel update: {new_data}")
            # Comparación profunda para detectar cualquier cambio relevante
            if prev_data is None or any(prev_data.get(cat, {}) != new_data.get(cat, {}) for cat in new_data):
                self.data = new_data
                self._dirty = True
                self.event_system.notify("stats_updated", self.data)
            else:
                self._dirty = False

    def get_stats(self):
        with self._lock:
            return copy.deepcopy(self.data)

    def is_dirty(self):
        return self._dirty

    def clean(self):
        self._dirty = False

    def get_last_record_game(self):
        """Devuelve el número del juego donde se obtuvo el último récord, usando el agente si está disponible."""
        if self.agent and hasattr(self.agent, 'last_record_game'):
            return getattr(self.agent, 'last_record_game', 0)
        return 0
