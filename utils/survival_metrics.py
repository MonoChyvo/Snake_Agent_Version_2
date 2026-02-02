"""
Módulo de métricas de supervivencia para el proyecto Snake DQN.

Este módulo proporciona herramientas para rastrear y analizar métricas enfocadas
en la supervivencia del agente, ignorando métricas que pueden engañar durante
las primeras fases de entrenamiento.

Métricas principales:
- Pasos promedio por episodio
- Frecuencia de muerte por colisión inmediata
- Número de episodios donde no muere en los primeros N pasos
- Comidas ocasionales (aunque sean pocas)

Versión: 1.0.0
"""

import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class EpisodeData:
    """Datos de un episodio individual"""

    episode_num: int
    steps: int
    score: int
    died_early: bool  # Murió en primeros N pasos
    timestamp: str
    cause_of_death: str = "collision"  # collision, timeout, etc.


class SurvivalMetricsTracker:
    """
    Rastreador de métricas de supervivencia.

    Mantiene un historial de episodios y calcula métricas enfocadas en supervivencia.
    """

    def __init__(self, window_size: int = 100, early_death_threshold: int = 10):
        """
        Inicializa el tracker.

        Args:
            window_size: Tamaño de la ventana para promedios móviles
            early_death_threshold: Número de pasos para considerar muerte inmediata
        """
        self.window_size = window_size
        self.early_death_threshold = early_death_threshold

        # Historial completo de episodios
        self.episodes: List[EpisodeData] = []

        # Ventanas deslizantes para cálculos eficientes
        self.recent_steps = deque(maxlen=window_size)
        self.recent_scores = deque(maxlen=window_size)
        self.recent_early_deaths = deque(maxlen=window_size)

    def record_episode(
        self,
        episode_num: int,
        steps: int,
        score: int,
        cause_of_death: str = "collision",
    ) -> None:
        """
        Registra un episodio completado.

        Args:
            episode_num: Número del episodio
            steps: Número de pasos sobrevividos
            score: Puntuación obtenida (comidas)
            cause_of_death: Causa de muerte (collision, timeout, etc.)
        """
        died_early = steps <= self.early_death_threshold

        episode = EpisodeData(
            episode_num=episode_num,
            steps=steps,
            score=score,
            died_early=died_early,
            timestamp=datetime.now().isoformat(),
            cause_of_death=cause_of_death,
        )

        self.episodes.append(episode)
        self.recent_steps.append(steps)
        self.recent_scores.append(score)
        self.recent_early_deaths.append(died_early)

    def get_avg_steps(self, window: Optional[int] = None) -> float:
        """
        Calcula pasos promedio por episodio.

        Args:
            window: Tamaño de ventana (None = usar ventana configurada)

        Returns:
            Pasos promedio
        """
        if window is None:
            steps_list = list(self.recent_steps)
        else:
            steps_list = [ep.steps for ep in self.episodes[-window:]]

        return np.mean(steps_list) if steps_list else 0.0

    def get_immediate_death_rate(self, window: Optional[int] = None) -> float:
        """
        Calcula la tasa de muerte inmediata (%).

        Args:
            window: Tamaño de ventana (None = usar ventana configurada)

        Returns:
            Porcentaje de muertes inmediatas (0.0 a 1.0)
        """
        if window is None:
            early_deaths = list(self.recent_early_deaths)
        else:
            early_deaths = [ep.died_early for ep in self.episodes[-window:]]

        if not early_deaths:
            return 0.0

        return sum(early_deaths) / len(early_deaths)

    def count_survival_episodes(
        self, min_steps: int = 50, window: Optional[int] = None
    ) -> int:
        """
        Cuenta episodios donde sobrevivió al menos min_steps pasos.

        Args:
            min_steps: Mínimo de pasos para considerar éxito
            window: Tamaño de ventana (None = usar ventana configurada)

        Returns:
            Número de episodios exitosos
        """
        if window is None:
            episodes_to_check = self.episodes[-self.window_size :]
        else:
            episodes_to_check = self.episodes[-window:]

        return sum(1 for ep in episodes_to_check if ep.steps >= min_steps)

    def get_food_frequency(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        Calcula estadísticas de comidas ocasionales.

        Args:
            window: Tamaño de ventana (None = usar ventana configurada)

        Returns:
            Diccionario con estadísticas de comidas
        """
        if window is None:
            scores = list(self.recent_scores)
        else:
            scores = [ep.score for ep in self.episodes[-window:]]

        if not scores:
            return {
                "avg_food": 0.0,
                "max_food": 0,
                "episodes_with_food": 0,
                "food_rate": 0.0,
            }

        return {
            "avg_food": np.mean(scores),
            "max_food": max(scores),
            "episodes_with_food": sum(1 for s in scores if s > 0),
            "food_rate": sum(1 for s in scores if s > 0) / len(scores),
        }

    def get_summary(self, window: Optional[int] = None) -> Dict:
        """
        Obtiene un resumen completo de métricas de supervivencia.

        Args:
            window: Tamaño de ventana (None = usar ventana configurada)

        Returns:
            Diccionario con todas las métricas
        """
        food_stats = self.get_food_frequency(window)

        return {
            "total_episodes": len(self.episodes),
            "avg_steps": self.get_avg_steps(window),
            "immediate_death_rate": self.get_immediate_death_rate(window),
            "survival_episodes": self.count_survival_episodes(window=window),
            "food_stats": food_stats,
            "window_size": window or self.window_size,
        }

    def get_trend(self, metric: str = "steps", periods: int = 2) -> str:
        """
        Detecta tendencia en una métrica (mejorando/empeorando/estable).

        Args:
            metric: Métrica a analizar ('steps', 'death_rate', 'food')
            periods: Número de períodos a comparar

        Returns:
            'improving', 'worsening', 'stable'
        """
        if len(self.episodes) < self.window_size * periods:
            return "insufficient_data"

        # Dividir en períodos
        period_size = self.window_size
        recent_period = self.episodes[-period_size:]
        previous_period = self.episodes[-2 * period_size : -period_size]

        if metric == "steps":
            recent_avg = np.mean([ep.steps for ep in recent_period])
            previous_avg = np.mean([ep.steps for ep in previous_period])
            threshold = 5  # Diferencia mínima para considerar cambio

            if recent_avg > previous_avg + threshold:
                return "improving"
            elif recent_avg < previous_avg - threshold:
                return "worsening"
            else:
                return "stable"

        elif metric == "death_rate":
            recent_rate = sum(ep.died_early for ep in recent_period) / len(
                recent_period
            )
            previous_rate = sum(ep.died_early for ep in previous_period) / len(
                previous_period
            )
            threshold = 0.1  # 10% de diferencia

            if recent_rate < previous_rate - threshold:
                return "improving"
            elif recent_rate > previous_rate + threshold:
                return "worsening"
            else:
                return "stable"

        elif metric == "food":
            recent_avg = np.mean([ep.score for ep in recent_period])
            previous_avg = np.mean([ep.score for ep in previous_period])
            threshold = 0.5

            if recent_avg > previous_avg + threshold:
                return "improving"
            elif recent_avg < previous_avg - threshold:
                return "worsening"
            else:
                return "stable"

        return "unknown"

    def export_to_json(self, filepath: str) -> None:
        """
        Exporta el historial de episodios a JSON.

        Args:
            filepath: Ruta del archivo de salida
        """
        data = {
            "config": {
                "window_size": self.window_size,
                "early_death_threshold": self.early_death_threshold,
            },
            "episodes": [
                {
                    "episode_num": ep.episode_num,
                    "steps": ep.steps,
                    "score": ep.score,
                    "died_early": ep.died_early,
                    "timestamp": ep.timestamp,
                    "cause_of_death": ep.cause_of_death,
                }
                for ep in self.episodes
            ],
            "summary": self.get_summary(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, filepath: str) -> None:
        """
        Carga historial de episodios desde JSON.

        Args:
            filepath: Ruta del archivo de entrada
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Restaurar configuración
        config = data.get("config", {})
        self.window_size = config.get("window_size", self.window_size)
        self.early_death_threshold = config.get(
            "early_death_threshold", self.early_death_threshold
        )

        # Restaurar episodios
        self.episodes = []
        self.recent_steps.clear()
        self.recent_scores.clear()
        self.recent_early_deaths.clear()

        for ep_data in data.get("episodes", []):
            episode = EpisodeData(
                episode_num=ep_data["episode_num"],
                steps=ep_data["steps"],
                score=ep_data["score"],
                died_early=ep_data["died_early"],
                timestamp=ep_data["timestamp"],
                cause_of_death=ep_data.get("cause_of_death", "collision"),
            )
            self.episodes.append(episode)

            # Actualizar ventanas deslizantes
            self.recent_steps.append(episode.steps)
            self.recent_scores.append(episode.score)
            self.recent_early_deaths.append(episode.died_early)


# Funciones auxiliares para análisis rápido


def calculate_avg_steps(episodes: List[EpisodeData], window: int = 100) -> float:
    """Calcula pasos promedio de una lista de episodios."""
    recent = episodes[-window:] if len(episodes) > window else episodes
    return np.mean([ep.steps for ep in recent]) if recent else 0.0


def calculate_immediate_death_rate(
    episodes: List[EpisodeData], threshold: int = 10, window: int = 100
) -> float:
    """Calcula tasa de muerte inmediata de una lista de episodios."""
    recent = episodes[-window:] if len(episodes) > window else episodes
    if not recent:
        return 0.0
    early_deaths = sum(1 for ep in recent if ep.steps <= threshold)
    return early_deaths / len(recent)


def count_survival_episodes(
    episodes: List[EpisodeData], min_steps: int = 50, window: int = 100
) -> int:
    """Cuenta episodios exitosos de una lista de episodios."""
    recent = episodes[-window:] if len(episodes) > window else episodes
    return sum(1 for ep in recent if ep.steps >= min_steps)


def calculate_food_frequency(
    episodes: List[EpisodeData], window: int = 100
) -> Dict[str, float]:
    """Calcula estadísticas de comidas de una lista de episodios."""
    recent = episodes[-window:] if len(episodes) > window else episodes
    if not recent:
        return {
            "avg_food": 0.0,
            "max_food": 0,
            "episodes_with_food": 0,
            "food_rate": 0.0,
        }

    scores = [ep.score for ep in recent]
    return {
        "avg_food": np.mean(scores),
        "max_food": max(scores),
        "episodes_with_food": sum(1 for s in scores if s > 0),
        "food_rate": sum(1 for s in scores if s > 0) / len(scores),
    }
