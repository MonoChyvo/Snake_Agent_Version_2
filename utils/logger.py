"""
Módulo de logging para el proyecto Snake DQN.
Proporciona funciones para mostrar información de manera estructurada
durante las diferentes fases del entrenamiento.
"""

import os
import numpy as np
from datetime import datetime
from colorama import Fore, Style, init

# Inicializar colorama para que funcione correctamente en Windows
init()

# Colores para diferentes tipos de mensajes
class LogColors:
    HEADER = Fore.LIGHTBLUE_EX
    GAME_NUMBER = Fore.LIGHTYELLOW_EX
    SUCCESS = Fore.GREEN
    INFO = Fore.CYAN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    CRITICAL = Fore.RED
    HIGHLIGHT = Fore.LIGHTMAGENTA_EX
    METRICS = Fore.MAGENTA
    SEPARATOR = Fore.RED
    RESET = Style.RESET_ALL

# Separadores
SEPARATOR_FULL = "=" * 60
SEPARATOR_SECTION = "-" * 50
SEPARATOR_SMALL = "-" * 25

class Logger:
    """Clase para manejar los logs del entrenamiento de manera estructurada"""

    @staticmethod
    def print_header(title):
        """Imprime un encabezado para una sección principal"""
        print("\n" + LogColors.SEPARATOR + SEPARATOR_FULL + LogColors.RESET)
        print(LogColors.HEADER + f"{title:^60}" + LogColors.RESET)
        print(LogColors.SEPARATOR + SEPARATOR_FULL + LogColors.RESET)

    @staticmethod
    def print_section(title):
        """Imprime un encabezado para una subsección"""
        print("\n" + LogColors.SEPARATOR + SEPARATOR_SECTION + LogColors.RESET)
        print(LogColors.INFO + f" {title} " + LogColors.RESET)
        print(LogColors.SEPARATOR + SEPARATOR_SECTION + LogColors.RESET)

    @staticmethod
    def print_game_header(game_number):
        """Imprime el encabezado de un juego"""
        print("\n" + LogColors.SEPARATOR + SEPARATOR_SECTION + LogColors.RESET)
        print(LogColors.GAME_NUMBER + f"{'|| Game ' + str(game_number) + ' ||':^50}" + LogColors.RESET)
        print(LogColors.SEPARATOR + SEPARATOR_SECTION + LogColors.RESET)

    @staticmethod
    def print_info(message):
        """Imprime un mensaje informativo"""
        print(LogColors.INFO + message + LogColors.RESET)

    @staticmethod
    def print_success(message):
        """Imprime un mensaje de éxito"""
        print(LogColors.SUCCESS + message + LogColors.RESET)

    @staticmethod
    def print_warning(message):
        """Imprime una advertencia"""
        print(LogColors.WARNING + "⚠️ " + message + LogColors.RESET)

    @staticmethod
    def print_error(message):
        """Imprime un error"""
        print(LogColors.ERROR + "❌ " + message + LogColors.RESET)

    @staticmethod
    def print_critical(message):
        """Imprime un error crítico"""
        print(LogColors.CRITICAL + "🚨 ALERTA CRÍTICA: " + message + LogColors.RESET)

    @staticmethod
    def print_metric(name, value, format_spec=""):
        """Imprime una métrica con formato"""
        if format_spec:
            formatted_value = f"{value:{format_spec}}"
        else:
            formatted_value = str(value)
        print(LogColors.INFO + f"{name}: " + LogColors.METRICS + formatted_value + LogColors.RESET)

    @staticmethod
    def print_metrics_group(title, metrics_dict, formats=None):
        """Imprime un grupo de métricas con un título"""
        if formats is None:
            formats = {}

        print(LogColors.INFO + f"\n{title}:" + LogColors.RESET)
        for key, value in metrics_dict.items():
            if value is not None:  # Solo imprimir si hay valor
                format_spec = formats.get(key, "")
                Logger.print_metric(f"  {key}", value, format_spec)

    @staticmethod
    def print_exploration_status(active, games_left=None):
        """Imprime el estado de la fase de exploración"""
        if active:
            print(LogColors.WARNING + f"En fase de exploración: {games_left} juegos restantes" + LogColors.RESET)
        else:
            print(LogColors.INFO + "Modo de explotación activo (pathfinding habilitado)" + LogColors.RESET)

    @staticmethod
    def print_memory_status(game_number, buffer_size_mb, experiences_count, system_memory_percent, buffer_fill_ratio=None):
        """Imprime el estado de la memoria"""
        print(LogColors.INFO + f"\nMemory Monitor - Game {game_number}:" + LogColors.RESET)
        print(f"  Replay Buffer: {buffer_size_mb:.2f} MB ({experiences_count} experiences)")
        print(f"  System Memory: {system_memory_percent:.1f}% used")

        if buffer_fill_ratio and buffer_fill_ratio > 0.8:
            print(LogColors.WARNING + f"  Buffer fill ratio: {buffer_fill_ratio:.2f}" + LogColors.RESET)

    @staticmethod
    def print_training_start():
        """Imprime información al inicio del entrenamiento"""
        Logger.print_header("SNAKE DQN - INICIO DE ENTRENAMIENTO")

    @staticmethod
    def print_training_end():
        """Imprime información al finalizar el entrenamiento"""
        Logger.print_header("ENTRENAMIENTO COMPLETADO")
        Logger.print_success("El entrenamiento ha finalizado correctamente.")

    @staticmethod
    def print_evaluation_header(is_periodic=True):
        """Imprime el encabezado de una evaluación"""
        if is_periodic:
            Logger.print_section("EVALUACIÓN PERIÓDICA DEL AGENTE")
        else:
            Logger.print_section("EVALUACIÓN FINAL DEL AGENTE")

    @staticmethod
    def print_evaluation_results(results):
        """Imprime los resultados de la evaluación de manera formateada"""
        print(f"Episodios evaluados: {results['episodes']}")
        print(f"Puntuación media: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
        print(f"Puntuación máxima: {results['max_score']}")
        print(f"Puntuación mínima: {results['min_score']}")
        print(f"Longitud media: {results['mean_length']:.2f}")
        print(f"Pasos medios: {results['mean_steps']:.2f}")
        print(f"Recompensa media: {results['mean_reward']:.2f}")

    @staticmethod
    def print_model_saved(path, timestamp=None):
        """Imprime información sobre el guardado del modelo"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        Logger.print_info(f"Modelo guardado: {timestamp}")

    @staticmethod
    def print_weight_norms(norms_dict):
        """Imprime las normas de los pesos de manera formateada"""
        Logger.print_metrics_group("Normas de pesos", norms_dict, formats={"w1_norm": ".4f", "w2_norm": ".4f", "w3_norm": ".4f"})

    @staticmethod
    def print_game_summary(game_stats):
        """Imprime un resumen del juego actual"""
        # Métricas básicas
        basic_metrics = {
            "Score": game_stats.get("score"),
            "Pasos": game_stats.get("steps"),
            "Récord": game_stats.get("record")
        }
        Logger.print_metrics_group("Métricas básicas", basic_metrics)

        # Métricas de recompensa
        reward_metrics = {
            "Recompensa total": game_stats.get("total_reward"),
            "Recompensa media": game_stats.get("avg_reward"),
            "Último récord (juego)": game_stats.get("last_record_game")
        }
        Logger.print_metrics_group("Recompensas", reward_metrics,
                                  formats={"Recompensa total": ".4f", "Recompensa media": ".2f"})

        # Métricas de eficiencia
        efficiency_metrics = {
            "Ratio de eficiencia": game_stats.get("efficiency_ratio"),
            "Pasos por comida": game_stats.get("steps_per_food")
        }
        Logger.print_metrics_group("Eficiencia", efficiency_metrics,
                                  formats={"Ratio de eficiencia": ".2f", "Pasos por comida": ".2f"})

        # Métricas del modelo
        model_metrics = {
            "Pérdida": game_stats.get("loss"),
            "Temperatura": game_stats.get("temperature"),
            "Learning rate": game_stats.get("learning_rate")
        }
        Logger.print_metrics_group("Modelo", model_metrics,
                                  formats={"Pérdida": ".4f", "Temperatura": ".4f", "Learning rate": ".6f"})
