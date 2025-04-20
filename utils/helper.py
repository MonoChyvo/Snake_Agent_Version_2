

import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style
import numpy as np
import time
from utils.config import ALERT_THRESHOLDS

# Importar el sistema de registro de errores
from utils.error_logger import event_system_logger, log_error, log_warning, log_info

# Sistema de eventos mejorado para notificar cambios en los datos
class EventSystem:
    def __init__(self):
        self.listeners = {}
        self.data_cache = {}
        self.last_update_time = {}
        self.event_counts = {}  # Contador de eventos para estadísticas
        log_info(event_system_logger, "EventSystem", "Sistema de eventos inicializado")

    def register_listener(self, event_name, callback):
        """Registra un listener para un tipo de evento específico."""
        if event_name not in self.listeners:
            self.listeners[event_name] = []

        # Evitar duplicados
        if callback not in self.listeners[event_name]:
            self.listeners[event_name].append(callback)
            log_info(event_system_logger, "EventSystem", f"Listener registrado para evento '{event_name}'")
        else:
            log_warning(event_system_logger, "EventSystem", f"Intento de registrar un listener duplicado para '{event_name}'")

    def notify(self, event_name, data=None):
        """Notifica a todos los listeners registrados para un evento específico."""
        # Actualizar cache y timestamp para todos los eventos, incluso si no hay listeners
        self.data_cache[event_name] = data
        self.last_update_time[event_name] = time.time()

        # Actualizar contador de eventos
        if event_name not in self.event_counts:
            self.event_counts[event_name] = 0
        self.event_counts[event_name] += 1

        # Notificar a todos los listeners si existen
        if event_name in self.listeners:
            for callback in self.listeners[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    # Registrar el error en lugar de ignorarlo silenciosamente
                    log_error(
                        event_system_logger,
                        "EventSystem",
                        f"Error al notificar evento '{event_name}'",
                        exception=e,
                        context={"data": str(data)[:100] if data else None}  # Limitar longitud para evitar logs enormes
                    )

    def get_cached_data(self, event_name):
        """Obtiene los datos más recientes para un tipo de evento específico."""
        return self.data_cache.get(event_name, None)

    def get_last_update_time(self, event_name):
        """Obtiene el timestamp de la última actualización para un tipo de evento específico."""
        return self.last_update_time.get(event_name, 0)

    def get_event_stats(self):
        """Obtiene estadísticas sobre los eventos procesados."""
        return {
            "event_counts": self.event_counts,
            "registered_events": list(self.listeners.keys()),
            "cached_events": list(self.data_cache.keys())
        }

# Crear instancia global del sistema de eventos
event_system = EventSystem()

# Variable global para almacenar las normas de pesos más recientes
latest_weight_norms = {
    "Capa_linear_01": 0.0,
    "Capa_linear_02": 0.0,
    "Capa_linear_03": 0.0
}

# Variable global para almacenar el resumen del juego más reciente
latest_game_summary = {
    "Último récord obtenido en partida": 0,
    "Puntuaciones recientes": "[]",
    "Recompensa": 0.0,
    "Puntuación": 0,
    "Récord": 0
}

# Variable global para almacenar los parámetros del modelo más recientes
latest_model_params = {
    "loss": 0.0,
    "temperature": 0.99,
    "pathfinding_enabled": True,
    "exploration_phase": False,
    "exploration_games_left": 0,
    "learning_rate": 0.001
}

# Banderas para indicar si los datos han sido actualizados
weight_norms_updated = False
game_summary_updated = False
model_params_updated = False

# Timestamps de las últimas actualizaciones
last_weight_norms_update = 0
last_game_summary_update = 0
last_model_params_update = 0

# Función auxiliar para obtener las normas de pesos más recientes
def get_weight_norms(force_update=False):
    """Obtiene las normas de pesos más recientes.

    Args:
        force_update: Si es True, fuerza una actualización de las normas de pesos.

    Returns:
        Un diccionario con las normas de pesos más recientes.
    """
    global latest_weight_norms, weight_norms_updated, last_weight_norms_update, event_system

    # Verificar si hay datos en el cache del sistema de eventos
    cached_data = event_system.get_cached_data("weight_norms_updated")
    if cached_data is not None:
        # Usar los datos del cache sin imprimir mensajes de depuración
        return cached_data

    # Si no hay datos en el cache o se fuerza la actualización, intentar obtener los datos
    if force_update or not weight_norms_updated:
        # Intentar obtener el agente global
        agent = globals().get("agent", None)
        if agent and hasattr(agent, "model"):
            try:
                # Llamar a print_weight_norms para actualizar los datos sin mensajes de depuración
                return print_weight_norms(agent)
            except Exception:
                # Ignorar errores silenciosamente para no saturar la consola
                pass

    # Si no se pudo actualizar, devolver los valores actuales
    return latest_weight_norms


# Función auxiliar para obtener los parámetros del modelo más recientes
def get_model_params(force_update=False):
    """Obtiene los parámetros del modelo más recientes.

    Args:
        force_update: Si es True, fuerza una actualización de los parámetros del modelo.

    Returns:
        Un diccionario con los parámetros del modelo más recientes.
    """
    global latest_model_params, model_params_updated, last_model_params_update, event_system

    # Verificar si hay datos en el cache del sistema de eventos
    cached_data = event_system.get_cached_data("model_params_updated")
    if cached_data is not None:
        # Actualizar la variable global con los datos del cache
        latest_model_params.update(cached_data)
        model_params_updated = True
        last_model_params_update = time.time()
        return cached_data

    # Si no hay datos en el cache o se fuerza la actualización, intentar obtener los datos
    if force_update or not model_params_updated:
        # Intentar obtener el agente global
        agent = globals().get("agent", None)
        if agent:
            try:
                # Obtener los parámetros del modelo directamente del agente
                model_params = {
                    "loss": agent.last_loss if hasattr(agent, "last_loss") else 0.0,
                    "temperature": agent.temperature if hasattr(agent, "temperature") else 0.99,
                    "pathfinding_enabled": agent.pathfinding_enabled if hasattr(agent, "pathfinding_enabled") else True,
                    "exploration_phase": agent.exploration_phase if hasattr(agent, "exploration_phase") else False,
                    "exploration_games_left": agent.exploration_games_left if hasattr(agent, "exploration_games_left") else 0
                }

                # Obtener learning rate si está disponible
                if hasattr(agent, "trainer") and hasattr(agent.trainer, "optimizer"):
                    for param_group in agent.trainer.optimizer.param_groups:
                        if 'lr' in param_group:
                            model_params["learning_rate"] = param_group['lr']
                            break

                # Actualizar la variable global
                latest_model_params.update(model_params)
                model_params_updated = True
                last_model_params_update = time.time()

                # Notificar a través del sistema de eventos
                event_system.notify("model_params_updated", model_params)

                return model_params
            except Exception:
                # Ignorar errores silenciosamente para no saturar la consola
                pass

    # Si no se pudo actualizar, devolver los valores actuales
    return latest_model_params

def plot_training_progress(scores, mean_scores, save_plot=False, save_path="plots", filename="training_progress.png"):

    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Scores")
    plt.plot(mean_scores, label="Mean Scores")
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    if save_plot:
        try:
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path)
            # No imprimir mensajes para no saturar la consola
            pass
        except Exception:
            # Ignorar errores silenciosamente para no saturar la consola
            pass

    plt.show(block=False)
    plt.pause(0.1)


def log_game_results(agent, score, record, game, avg_loss=None):
    timestamp = datetime.now()

    # Calculate efficiency metrics
    snake_length = len(game.snake) if hasattr(game, 'snake') else score + 1
    unique_positions = len(set((p.x, p.y) for p in game.snake)) if hasattr(game, 'snake') else 0
    efficiency_ratio = unique_positions / snake_length if snake_length > 0 else 0

    # Calculate reward statistics
    total_reward = sum(game.reward_history) if hasattr(game, 'reward_history') else 0
    avg_reward = total_reward / len(game.reward_history) if hasattr(game, 'reward_history') and len(game.reward_history) > 0 else 0

    # Calculate game duration and speed
    steps = game.steps if hasattr(game, 'steps') else 0
    steps_per_food = steps / score if score > 0 else steps

    # Calculate action distribution if available
    from utils.numpy_utils import safe_mean

    action_distribution = {}
    if hasattr(game, 'action_history') and len(game.action_history) > 0:
        actions = np.array(game.action_history)
        action_distribution = {
            'straight_pct': safe_mean(actions == 0) * 100,
            'right_pct': safe_mean(actions == 1) * 100,
            'left_pct': safe_mean(actions == 2) * 100,
        }

    # Get model weight statistics if available
    weight_stats = {}
    if hasattr(agent, 'model') and hasattr(agent.model, 'linear1'):
        try:
            weight_stats = {
                'w1_norm': agent.model.linear1.weight.data.norm().item(),
                'w2_norm': agent.model.linear2.weight.data.norm().item(),
                'w3_norm': agent.model.linear3.weight.data.norm().item()
            }

            # Añadir ratios de pesos para monitoreo
            weight_stats['w2_w1_ratio'] = weight_stats['w2_norm'] / weight_stats['w1_norm'] if weight_stats['w1_norm'] > 0 else 0
            weight_stats['w3_w1_ratio'] = weight_stats['w3_norm'] / weight_stats['w1_norm'] if weight_stats['w1_norm'] > 0 else 0
        except Exception:
            # Ignorar errores silenciosamente para no saturar la consola
            pass

    # Create comprehensive game result entry
    game_result = {
        'game_number': agent.n_games,
        'score': score,
        'snake_length': snake_length,
        'record': record,
        'temperature': agent.temperature,
        'steps_taken': steps,
        'total_reward': total_reward,
        'steps_per_food': steps_per_food,
        'timestamp': timestamp,
        'avg_reward': avg_reward,
        'efficiency_ratio': efficiency_ratio,
        'loss': avg_loss,

        # Advanced metrics
        "exploration_rate": agent.epsilon if hasattr(agent, 'epsilon') else None,

        # Environment Understanding
        "food_distance_stats": {
            "mean": safe_mean(game.food_distances) if hasattr(game, 'food_distances') else None,
            "initial_vs_final_ratio": compute_distance_progress(game.food_distances) if hasattr(game, 'food_distances') else None
        },
        "open_space_ratio": game.avg_open_space_ratio if hasattr(game, 'avg_open_space_ratio') else None,

        # Temporal Performance
        "avg_decision_time": safe_mean(game.decision_times) if hasattr(game, 'decision_times') else None,
        "game_duration_seconds": game.game_duration if hasattr(game, 'game_duration') else None,
        "performance_improvement": calculate_improvement_rate(agent) if hasattr(agent, 'recent_scores') else None,
    }

    # Add action distribution if available
    if action_distribution:
        game_result.update(action_distribution)

    # Add weight stats if available
    if weight_stats:
        game_result.update(weight_stats)

    # Add to game results and save periodically
    agent.game_results.append(game_result)

    # Check metrics for alerts
    check_metrics_alerts(game_result)

    if agent.n_games % 1 == 0:
        df_game_results = pd.DataFrame(agent.game_results)
        os.makedirs("results", exist_ok=True)

        # Create summary with rolling averages for key metrics
        if len(df_game_results) >= 10:  # Only if we have enough data
            summary_cols = ["game_number", "score", "steps_taken", "total_reward", "avg_reward", "efficiency_ratio"]
            available_cols = [col for col in summary_cols if col in df_game_results.columns]

            # Add rolling averages directly to the main dataframe
            for col in available_cols:
                if col != "game_number":
                    df_game_results[f"{col}_avg10"] = df_game_results[col].rolling(10).mean()
                    df_game_results[f"{col}_avg50"] = df_game_results[col].rolling(50).mean()

        # Save the single unified CSV file
        save_game_results(agent, df_game_results)

        # Generate analysis plots
        if len(df_game_results) >= 50:
            create_analysis_plots(df_game_results, "results/analysis_plots")


def create_analysis_plots(df, save_path="plots/analysis"):
    """Create detailed analysis plots from game results data."""
    os.makedirs(save_path, exist_ok=True)

    weight_cols = ['w1_norm', 'w2_norm', 'w3_norm']

    # 1. Score progression with rolling averages
    if all(col in df.columns for col in weight_cols):
        plt.figure(figsize=(12, 6))
        # Calcular ratios de normas
        df['w2_w1_ratio'] = df['w2_norm'] / df['w1_norm']
        df['w3_w1_ratio'] = df['w3_norm'] / df['w1_norm']

        plt.plot(df['game_number'], df['w2_w1_ratio'], color='purple', label='w2/w1 ratio')
        plt.plot(df['game_number'], df['w3_w1_ratio'], color='orange', label='w3/w1 ratio')
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        plt.title('Weight Norm Ratios (Ideal: closer to 1.0)')
        plt.xlabel('Game Number')
        plt.ylabel('Ratio')
        plt.legend()
        plt.savefig(f"{save_path}/weight_norm_ratios.png")
        plt.close()

    # 2. Reward analysis
    if 'avg_reward' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['game_number'], df['avg_reward'], color='orange', alpha=0.3, label='Average Reward')
        plt.plot(df['game_number'], df['avg_reward'].rolling(50).mean(), color='red', label='Avg Reward (50-game)')
        plt.title('Reward Analysis')
        plt.xlabel('Game Number')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.savefig(f"{save_path}/reward_analysis.png")
        plt.close()

    # 3. Efficiency metrics
    if 'efficiency_ratio' in df.columns and 'steps_per_food' in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Efficiency ratio plot
        ax1.plot(df['game_number'], df['efficiency_ratio'], color='green', alpha=0.3)
        ax1.plot(df['game_number'], df['efficiency_ratio'].rolling(30).mean(), color='darkgreen', linewidth=2)
        ax1.set_title('Movement Efficiency')
        ax1.set_ylabel('Efficiency Ratio')

        # Steps per food plot
        ax2.plot(df['game_number'], df['steps_per_food'], color='purple', alpha=0.3)
        ax2.plot(df['game_number'], df['steps_per_food'].rolling(30).mean(), color='darkviolet', linewidth=2)
        ax2.set_title('Steps Per Food')
        ax2.set_xlabel('Game Number')
        ax2.set_ylabel('Steps')

        plt.tight_layout()
        plt.savefig(f"{save_path}/efficiency_metrics.png")
        plt.close()

    # 4. Training metrics
    if 'avg_training_loss' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['game_number'], df['avg_training_loss'], color='blue', alpha=0.3, label='Training Loss')
        plt.plot(df['game_number'], df['avg_training_loss'].rolling(50).mean(), color='darkblue', linewidth=2, label='Avg Loss (50-game)')
        plt.title('Training Loss')
        plt.xlabel('Game Number')
        plt.ylabel('Loss')
        plt.yscale('log')  # Log scale often better for loss
        plt.legend()
        plt.savefig(f"{save_path}/training_loss.png")
        plt.close()

    # 5. Model weight norms (if available)
    if all(col in df.columns for col in weight_cols):
        plt.figure(figsize=(12, 6))
        for col, color in zip(weight_cols, ['red', 'green', 'blue']):
            plt.plot(df['game_number'], df[col], alpha=0.3, color=color, label=f'{col} raw')
            plt.plot(df['game_number'], df[col].rolling(50).mean(), color=color, linewidth=2, label=f'{col} avg')
        plt.title('Model Weight Norms')
        plt.xlabel('Game Number')
        plt.ylabel('Norm')
        plt.legend()
        plt.savefig(f"{save_path}/weight_norms.png")
        plt.close()


def update_plots(agent, score, total_score, plot_scores, plot_mean_scores):
    # Actualiza las listas con el nuevo score
    plot_scores.append(score)
    total_score += score
    mean_score = total_score / agent.n_games
    plot_mean_scores.append(mean_score)

    # Para un plot acumulativo, podrías cargar la data existente del CSV
    # y combinarla con los datos de la sesión actual si es necesario.

    # Actualizamos el plot cada 100 juegos, por ejemplo:
    if agent.n_games % 1 == 0:
        plt.figure(figsize=(10, 6))
        plt.title("Training Progress")
        plt.xlabel("Number of Games")
        plt.ylabel("Score")
        plt.plot(plot_scores, label="Scores")
        plt.plot(plot_mean_scores, label="Mean Scores")
        plt.ylim(ymin=0)
        plt.legend()
        plt.text(len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1]))
        plt.text(len(plot_mean_scores) - 1, plot_mean_scores[-1], str(plot_mean_scores[-1]))

        # Guarda el plot en un único archivo que se vaya actualizando
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/training_progress.png")
        plt.close()  # Cierra la figura para liberar memoria

    return total_score



def save_checkpoint(agent, loss, filename="model_MARK_IX.pth"):
    record_value = agent.record if hasattr(agent, 'record') else 0
    temperature_value = agent.temperature if hasattr(agent, 'temperature') else None

    agent.model.save(
        filename,
        n_games=agent.n_games,
        optimizer=agent.trainer.optimizer.state_dict(),
        loss=loss,
        last_record_game=agent.last_record_game,
        record=record_value,
        temperature=temperature_value
    )


def print_game_info(reward, score, last_record_game, record, recent_scores):
    # Imprime información del juego usando el nuevo sistema de logging
    from utils.logger import Logger

    game_stats = {
        "score": score,
        "record": record,
        "total_reward": reward,
        "last_record_game": last_record_game,
        "recent_scores": recent_scores
    }

    # Se ha eliminado la sección de resumen del juego como solicitado
    # Los valores importantes se muestran en otras secciones del panel


# Importar el sistema de registro de errores para este método
from utils.error_logger import data_update_logger, log_error, log_warning, log_info

def update_game_summary(game=None, agent=None, force_update=False):
    """Actualiza y obtiene el resumen del juego más reciente.

    Args:
        game (Game): Instancia del juego. Si es None, intenta obtener la instancia global.
        agent (Agent): Instancia del agente. Si es None, intenta obtener la instancia global.
        force_update (bool): Si es True, fuerza una actualización del resumen.

    Returns:
        dict: Un diccionario con el resumen del juego más reciente.
    """
    global game_summary_updated, last_game_summary_update, latest_game_summary, event_system

    # Verificar si hay datos en el cache del sistema de eventos
    cached_data = event_system.get_cached_data("game_summary_updated")
    if cached_data is not None and not force_update:
        log_info(data_update_logger, "GameSummary", "Usando datos en caché para el resumen del juego")
        return cached_data

    # Si no se proporcionaron instancias, intentar obtenerlas del contexto global
    if game is None:
        game = globals().get("game", None)
        if game is None:
            log_warning(data_update_logger, "GameSummary", "No se pudo obtener la instancia del juego del contexto global")

    if agent is None:
        agent = globals().get("agent", None)
        if agent is None:
            log_warning(data_update_logger, "GameSummary", "No se pudo obtener la instancia del agente del contexto global")

    # Si no tenemos las instancias necesarias, devolver los valores actuales
    if not game or not agent:
        log_warning(data_update_logger, "GameSummary", "Usando valores por defecto debido a falta de instancias")
        return latest_game_summary

    # Si tenemos las instancias necesarias, actualizar el resumen
    try:
        # Crear un resumen actualizado con valores formateados correctamente
        # Calcular la recompensa total con mayor precisión
        total_reward = 0
        if hasattr(game, "reward_history") and game.reward_history:
            total_reward = sum(game.reward_history)
            # Redondear a 4 decimales para mejor visualización
            total_reward = round(total_reward, 4)
            log_info(data_update_logger, "GameSummary", f"Recompensa total calculada: {total_reward}")
        else:
            log_warning(data_update_logger, "GameSummary", "No se pudo obtener el historial de recompensas")

        # Obtener la puntuación actual
        current_score = 0
        if hasattr(game, "score"):
            current_score = game.score
            log_info(data_update_logger, "GameSummary", f"Puntuación actual: {current_score}")
        else:
            log_warning(data_update_logger, "GameSummary", "No se pudo obtener la puntuación actual")

        # Obtener el valor de last_record_game directamente del agente
        last_record_game = 0
        if hasattr(agent, "last_record_game"):
            last_record_game = agent.last_record_game
            log_info(data_update_logger, "GameSummary", f"Último récord obtenido: {last_record_game}")
        else:
            log_warning(data_update_logger, "GameSummary", "El agente no tiene el atributo 'last_record_game'")

        # Obtener puntuaciones recientes
        recent_scores = []
        if hasattr(agent, "recent_scores"):
            if len(agent.recent_scores) >= 4:
                recent_scores = agent.recent_scores[-4:]
            else:
                recent_scores = agent.recent_scores
            log_info(data_update_logger, "GameSummary", f"Puntuaciones recientes obtenidas: {len(recent_scores)} elementos")
        else:
            log_warning(data_update_logger, "GameSummary", "El agente no tiene el atributo 'recent_scores'")

        # Crear resumen del juego
        summary = {
            "Último récord obtenido en partida": last_record_game,
            "Puntuaciones recientes": str(recent_scores),
            "Recompensa": total_reward,
            "Puntuación": current_score,
            "Récord": game.record if hasattr(game, "record") else 0
        }

        # Actualizar las variables globales
        latest_game_summary.update(summary)
        game_summary_updated = True
        last_game_summary_update = time.time()

        # Notificar a los listeners a través del sistema de eventos
        event_system.notify("game_summary_updated", summary)
        log_info(data_update_logger, "GameSummary", "Resumen del juego actualizado y notificado correctamente")

        return summary
    except Exception as e:
        # Registrar el error en lugar de ignorarlo silenciosamente
        log_error(
            data_update_logger,
            "GameSummary",
            "Error al actualizar el resumen del juego",
            exception=e,
            context={
                "game_has_reward_history": hasattr(game, "reward_history"),
                "game_has_score": hasattr(game, "score"),
                "game_has_record": hasattr(game, "record"),
                "agent_has_last_record_game": hasattr(agent, "last_record_game"),
                "agent_has_recent_scores": hasattr(agent, "recent_scores")
            }
        )

    # Si no se pudo actualizar, devolver los valores actuales
    return latest_game_summary


# Importar el sistema de registro de errores para este método
from utils.error_logger import data_update_logger, log_error, log_warning, log_info

def print_weight_norms(agent):
    """Calcula y muestra las normas de los pesos para dar seguimiento al entrenamiento.
    Actualiza la variable global y notifica a los listeners a través del sistema de eventos.

    Args:
        agent: Instancia del agente con el modelo cuyos pesos se quieren analizar.

    Returns:
        dict: Un diccionario con las normas de los pesos de cada capa.
    """
    from utils.logger import Logger
    global latest_weight_norms, weight_norms_updated, last_weight_norms_update, event_system

    # Verificar que el agente es válido
    if agent is None:
        log_warning(data_update_logger, "WeightNorms", "Agente no válido proporcionado")
        return latest_weight_norms

    # Verificar que el agente tiene un modelo
    if not hasattr(agent, "model"):
        log_warning(data_update_logger, "WeightNorms", "El agente no tiene un modelo")
        return latest_weight_norms

    # Método 1: Acceso directo a los atributos del modelo
    try:
        # Verificar que el modelo tiene las capas esperadas
        if not hasattr(agent.model, "linear1") or not hasattr(agent.model, "linear2") or not hasattr(agent.model, "linear3"):
            raise AttributeError("El modelo no tiene las capas lineales esperadas")

        # Obtener las normas de los pesos
        w1_norm = float(agent.model.linear1.weight.data.norm().item())
        w2_norm = float(agent.model.linear2.weight.data.norm().item())
        w3_norm = float(agent.model.linear3.weight.data.norm().item())

        # Crear diccionario con los valores
        norms_dict = {
            "Capa_linear_01": w1_norm,
            "Capa_linear_02": w2_norm,
            "Capa_linear_03": w3_norm
        }

        # Actualizar la variable global con los valores más recientes
        latest_weight_norms.update(norms_dict)
        weight_norms_updated = True
        last_weight_norms_update = time.time()

        # Notificar a los listeners a través del sistema de eventos
        event_system.notify("weight_norms_updated", norms_dict)
        log_info(data_update_logger, "WeightNorms", "Normas de pesos calculadas y notificadas correctamente (Método 1)")

        # Imprimir en la consola usando el logger
        Logger.print_weight_norms(norms_dict)
        return norms_dict
    except Exception as e:
        log_warning(
            data_update_logger,
            "WeightNorms",
            "Error al calcular normas de pesos usando el Método 1, intentando Método 2",
            context={"error": str(e)}
        )

        # Método 2: Acceso a través del state_dict
        try:
            if not hasattr(agent.model, "state_dict"):
                raise AttributeError("El modelo no tiene método state_dict")

            state_dict = agent.model.state_dict()
            if "linear1.weight" not in state_dict or "linear2.weight" not in state_dict or "linear3.weight" not in state_dict:
                raise KeyError("El state_dict no contiene las claves esperadas para las capas lineales")

            w1_norm = float(state_dict["linear1.weight"].norm().item())
            w2_norm = float(state_dict["linear2.weight"].norm().item())
            w3_norm = float(state_dict["linear3.weight"].norm().item())

            norms_dict = {
                "Capa_linear_01": w1_norm,
                "Capa_linear_02": w2_norm,
                "Capa_linear_03": w3_norm
            }

            latest_weight_norms.update(norms_dict)
            weight_norms_updated = True
            last_weight_norms_update = time.time()

            event_system.notify("weight_norms_updated", norms_dict)
            log_info(data_update_logger, "WeightNorms", "Normas de pesos calculadas y notificadas correctamente (Método 2)")

            return norms_dict
        except Exception as e:
            # Registrar el error en lugar de ignorarlo silenciosamente
            log_error(
                data_update_logger,
                "WeightNorms",
                "Error al calcular normas de pesos",
                exception=e,
                context={
                    "agent_has_model": hasattr(agent, "model"),
                    "model_has_state_dict": hasattr(agent.model, "state_dict") if hasattr(agent, "model") else False
                }
            )

    # Si no se pudieron calcular las normas, devolver los valores actuales
    log_warning(data_update_logger, "WeightNorms", "Usando valores anteriores de normas de pesos")
    return latest_weight_norms


def save_game_results(agent, df_game_results):
    """
    Guarda los resultados de los juegos en un único archivo CSV centralizado, acumulando los datos.
    """
    # Asegurar que el directorio existe
    os.makedirs("results", exist_ok=True)
    csv_path = "results/MARK_IX_game_results.csv"

    # Si el archivo ya existe, cargarlo y concatenar los nuevos resultados
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_game_results], ignore_index=True)
        except Exception:
            # Ignorar errores silenciosamente para no saturar la consola
            df_combined = df_game_results
    else:
        df_combined = df_game_results

    # Guardar el CSV con los datos acumulados
    df_combined.to_csv(csv_path, index=False)



def compute_action_entropy(q_values_history):
    """
    Calculate the entropy of action selection to measure exploration vs exploitation balance.
    Higher entropy indicates more exploratory behavior.

    Args:
        q_values_history: List of Q-values arrays from each step

    Returns:
        float: Average entropy across all decision points
    """
    if not q_values_history or len(q_values_history) == 0:
        return None

    entropies = []
    for q_values in q_values_history:
        # Convert to probabilities using softmax
        q_values = np.array(q_values)
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probs = exp_q / np.sum(exp_q)

        # Calculate entropy: -sum(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(probs * np.log(probs + epsilon))
        entropies.append(entropy)

    return np.mean(entropies)


def compute_distance_progress(food_distances):
    """
    Analyze how efficiently the agent approaches food over the course of a game.
    Lower ratio indicates more direct paths to food.

    Args:
        food_distances: List of distances to food at each step

    Returns:
        float: Ratio of final distance to initial distance, or other progress metric
    """
    if not food_distances or len(food_distances) < 2:
        return None

    initial_distance = food_distances[0]
    final_distance = food_distances[-1]

    # Calculate ratio (if initial_distance is 0, avoid division by zero)
    if initial_distance == 0 or initial_distance < 0.001:
        return 0

    # Lower values are better - indicates closing distance to food
    return final_distance / initial_distance


def analyze_final_state(game):
    """
    Analyze the complexity of the game's final state before termination.
    Considers factors like available space, distances, and overall state complexity.

    Args:
        game: Game object with final state information

    Returns:
        float: Complexity score of the final state
    """
    complexity = 0.0

    # Check if we have the necessary attributes
    if not hasattr(game, 'final_state'):
        return None

    # Calculate various complexity metrics
    if hasattr(game, 'snake') and len(game.snake) > 0:
        # Calculate density - higher density means more complex state
        board_size = game.w * game.h if hasattr(game, 'w') and hasattr(game, 'h') else 400  # Default 20x20
        snake_density = len(game.snake) / board_size
        complexity += snake_density * 10  # Weight this component

    # Measure path complexity if available (more corners = more complex)
    if hasattr(game, 'action_history') and len(game.action_history) > 10:
        actions = np.array(game.action_history)
        changes = np.sum(np.abs(np.diff(actions)) > 0)  # Count direction changes
        change_rate = changes / (len(actions) - 1)
        complexity += change_rate * 5

    # Consider final reward signal if available
    if hasattr(game, 'reward_history') and len(game.reward_history) > 0:
        final_reward = game.reward_history[-1]
        # Negative rewards at end indicate more complex situations
        if final_reward < 0:
            complexity += abs(final_reward) * 2

    return complexity


def calculate_improvement_rate(agent):
    """
    Calculate how rapidly the agent's performance is improving.

    Args:
        agent: Agent object with performance history

    Returns:
        float: Rate of improvement as a percentage or score differential
    """
    from utils.numpy_utils import safe_mean

    if not hasattr(agent, 'recent_scores') or len(agent.recent_scores) < 20:
        return None

    # Compare average of last 10 games to average of 10 games before that
    recent_10 = agent.recent_scores[-10:]
    previous_10 = agent.recent_scores[-20:-10]

    if not previous_10:  # Not enough history
        return None

    recent_avg = safe_mean(recent_10)
    previous_avg = safe_mean(previous_10)

    # Avoid division by zero
    if abs(previous_avg) < 0.001:
        previous_avg = 0.1

    # Calculate improvement ratio
    improvement_ratio = (recent_avg - previous_avg) / abs(previous_avg)

    return improvement_ratio * 100  # Return as percentage


def check_metrics_alerts(metrics):
    """
    Verifica si alguna de las métricas ha superado los umbrales definidos y genera alertas.

    Args:
        metrics: Diccionario con las métricas a verificar
    """
    alerts = []

    # Verificar pérdida (loss)
    if 'loss' in metrics and metrics['loss'] is not None:
        loss_val = metrics['loss']
        if 'loss' in ALERT_THRESHOLDS:
            if 'critical' in ALERT_THRESHOLDS['loss'] and loss_val > ALERT_THRESHOLDS['loss']['critical']:
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': 'loss',
                    'value': loss_val,
                    'threshold': ALERT_THRESHOLDS['loss']['critical'],
                    'message': f"Pérdida crítica: {loss_val:.4f} > {ALERT_THRESHOLDS['loss']['critical']}"
                })
            elif 'high' in ALERT_THRESHOLDS['loss'] and loss_val > ALERT_THRESHOLDS['loss']['high']:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'loss',
                    'value': loss_val,
                    'threshold': ALERT_THRESHOLDS['loss']['high'],
                    'message': f"Pérdida alta: {loss_val:.4f} > {ALERT_THRESHOLDS['loss']['high']}"
                })

    # Verificar recompensa promedio
    if 'avg_reward' in metrics and metrics['avg_reward'] is not None:
        reward_val = metrics['avg_reward']
        if 'avg_reward' in ALERT_THRESHOLDS:
            if 'critical' in ALERT_THRESHOLDS['avg_reward'] and reward_val < ALERT_THRESHOLDS['avg_reward']['critical']:
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': 'avg_reward',
                    'value': reward_val,
                    'threshold': ALERT_THRESHOLDS['avg_reward']['critical'],
                    'message': f"Recompensa crítica: {reward_val:.4f} < {ALERT_THRESHOLDS['avg_reward']['critical']}"
                })
            elif 'low' in ALERT_THRESHOLDS['avg_reward'] and reward_val < ALERT_THRESHOLDS['avg_reward']['low']:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'avg_reward',
                    'value': reward_val,
                    'threshold': ALERT_THRESHOLDS['avg_reward']['low'],
                    'message': f"Recompensa baja: {reward_val:.4f} < {ALERT_THRESHOLDS['avg_reward']['low']}"
                })

    # Verificar ratio de eficiencia
    if 'efficiency_ratio' in metrics and metrics['efficiency_ratio'] is not None:
        efficiency_val = metrics['efficiency_ratio']
        if 'efficiency_ratio' in ALERT_THRESHOLDS:
            if 'low' in ALERT_THRESHOLDS['efficiency_ratio'] and efficiency_val < ALERT_THRESHOLDS['efficiency_ratio']['low']:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'efficiency_ratio',
                    'value': efficiency_val,
                    'threshold': ALERT_THRESHOLDS['efficiency_ratio']['low'],
                    'message': f"Eficiencia baja: {efficiency_val:.4f} < {ALERT_THRESHOLDS['efficiency_ratio']['low']}"
                })

    # Verificar pasos por comida
    if 'steps_per_food' in metrics and metrics['steps_per_food'] is not None:
        steps_val = metrics['steps_per_food']
        if 'steps_per_food' in ALERT_THRESHOLDS:
            if 'high' in ALERT_THRESHOLDS['steps_per_food'] and steps_val > ALERT_THRESHOLDS['steps_per_food']['high']:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'steps_per_food',
                    'value': steps_val,
                    'threshold': ALERT_THRESHOLDS['steps_per_food']['high'],
                    'message': f"Pasos por comida altos: {steps_val:.1f} > {ALERT_THRESHOLDS['steps_per_food']['high']}"
                })

    # Verificar ratio de normas de pesos
    if 'w2_w1_ratio' in metrics and metrics['w2_w1_ratio'] is not None:
        ratio_val = metrics['w2_w1_ratio']
        if 'weight_norm_ratio' in ALERT_THRESHOLDS:
            if 'critical' in ALERT_THRESHOLDS['weight_norm_ratio'] and ratio_val > ALERT_THRESHOLDS['weight_norm_ratio']['critical']:
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': 'weight_norm_ratio',
                    'value': ratio_val,
                    'threshold': ALERT_THRESHOLDS['weight_norm_ratio']['critical'],
                    'message': f"Ratio de pesos crítico: {ratio_val:.4f} > {ALERT_THRESHOLDS['weight_norm_ratio']['critical']}"
                })
            elif 'high' in ALERT_THRESHOLDS['weight_norm_ratio'] and ratio_val > ALERT_THRESHOLDS['weight_norm_ratio']['high']:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'weight_norm_ratio',
                    'value': ratio_val,
                    'threshold': ALERT_THRESHOLDS['weight_norm_ratio']['high'],
                    'message': f"Ratio de pesos alto: {ratio_val:.4f} > {ALERT_THRESHOLDS['weight_norm_ratio']['high']}"
                })

    # Mostrar las alertas generadas
    for alert in alerts:
        if alert['level'] == 'CRITICAL':
            print(Fore.RED + f"🚨 ALERTA CRÍTICA: {alert['message']}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + f"⚠️ ADVERTENCIA: {alert['message']}" + Style.RESET_ALL)