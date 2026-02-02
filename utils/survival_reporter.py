"""
Sistema de reportes de supervivencia para el proyecto Snake DQN.

Este módulo proporciona herramientas para generar reportes y visualizaciones
enfocadas en métricas de supervivencia, extrayendo datos de logs y del tracker.

Funcionalidades:
- Extracción de métricas desde logs
- Generación de reportes en formato markdown
- Visualizaciones de tendencias de supervivencia
- Comparación entre períodos

Versión: 1.0.0
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from utils.survival_metrics import (
    SurvivalMetricsTracker,
    EpisodeData,
    calculate_avg_steps,
    calculate_immediate_death_rate,
)


class SurvivalReporter:
    """
    Generador de reportes de supervivencia.

    Extrae métricas de logs y genera reportes con visualizaciones.
    """

    def __init__(self, tracker: Optional[SurvivalMetricsTracker] = None):
        """
        Inicializa el reporter.

        Args:
            tracker: Tracker de métricas (opcional, puede cargar desde logs)
        """
        self.tracker = tracker

    def parse_log_file(self, log_path: str) -> SurvivalMetricsTracker:
        """
        Extrae métricas de un archivo de log.

        Args:
            log_path: Ruta al archivo de log

        Returns:
            Tracker con métricas extraídas
        """
        from utils.config import SURVIVAL_METRICS

        tracker = SurvivalMetricsTracker(
            window_size=SURVIVAL_METRICS["survival_window"],
            early_death_threshold=SURVIVAL_METRICS["early_death_threshold"],
        )

        if not os.path.exists(log_path):
            print(f"⚠️ Archivo de log no encontrado: {log_path}")
            return tracker

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Patrones para extraer información
        # Ejemplo: "Juego #123 | Pasos: 45 | Puntuación: 3"
        game_pattern = re.compile(
            r"Juego #(\d+).*?Pasos:\s*(\d+).*?Puntuación:\s*(\d+)", re.IGNORECASE
        )

        for line in lines:
            match = game_pattern.search(line)
            if match:
                episode_num = int(match.group(1))
                steps = int(match.group(2))
                score = int(match.group(3))

                tracker.record_episode(episode_num, steps, score)

        return tracker

    def generate_report(self, output_path: str, include_plots: bool = True) -> str:
        """
        Genera un reporte completo de supervivencia en formato markdown.

        Args:
            output_path: Ruta del archivo de salida
            include_plots: Si True, genera y embebe gráficas

        Returns:
            Ruta del archivo generado
        """
        if self.tracker is None or len(self.tracker.episodes) == 0:
            print("⚠️ No hay datos para generar reporte")
            return ""

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Obtener métricas
        summary = self.tracker.get_summary()
        food_stats = summary["food_stats"]

        # Detectar tendencias
        steps_trend = self.tracker.get_trend("steps")
        death_trend = self.tracker.get_trend("death_rate")
        food_trend = self.tracker.get_trend("food")

        # Generar contenido del reporte
        report_lines = []
        report_lines.append("# Reporte de Supervivencia - Snake DQN\n")
        report_lines.append(
            f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        report_lines.append(f"**Total de episodios:** {summary['total_episodes']}\n")
        report_lines.append(
            f"**Ventana de análisis:** Últimos {summary['window_size']} episodios\n"
        )
        report_lines.append("\n---\n\n")

        # Sección: Métricas Clave
        report_lines.append("## 📊 Métricas Clave de Supervivencia\n\n")

        # Pasos promedio
        avg_steps = summary["avg_steps"]
        steps_emoji = (
            "📈"
            if steps_trend == "improving"
            else "📉" if steps_trend == "worsening" else "➡️"
        )
        report_lines.append(f"### {steps_emoji} Pasos Promedio por Episodio\n\n")
        report_lines.append(f"**{avg_steps:.1f} pasos**\n\n")
        report_lines.append(f"- Tendencia: **{self._format_trend(steps_trend)}**\n")

        # Muerte inmediata
        death_rate = summary["immediate_death_rate"]
        death_emoji = "✅" if death_rate < 0.3 else "⚠️" if death_rate < 0.6 else "🚨"
        report_lines.append(f"\n### {death_emoji} Frecuencia de Muerte Inmediata\n\n")
        report_lines.append(f"**{death_rate * 100:.1f}%** de los episodios\n\n")
        report_lines.append(f"- Tendencia: **{self._format_trend(death_trend)}**\n")

        # Episodios de supervivencia
        survival_count = summary["survival_episodes"]
        survival_rate = (
            survival_count / summary["window_size"] if summary["window_size"] > 0 else 0
        )
        survival_emoji = "🎯" if survival_rate > 0.5 else "🔄"
        report_lines.append(
            f"\n### {survival_emoji} Episodios de Supervivencia Exitosa\n\n"
        )
        report_lines.append(
            f"**{survival_count}** episodios con ≥50 pasos ({survival_rate * 100:.1f}%)\n\n"
        )

        # Comidas ocasionales
        food_emoji = "🍎" if food_stats["food_rate"] > 0.3 else "🌱"
        report_lines.append(f"\n### {food_emoji} Comidas Ocasionales\n\n")
        report_lines.append(
            f"- Promedio: **{food_stats['avg_food']:.2f}** comidas por episodio\n"
        )
        report_lines.append(f"- Máximo: **{food_stats['max_food']}** comidas\n")
        report_lines.append(
            f"- Episodios con comida: **{food_stats['episodes_with_food']}** ({food_stats['food_rate'] * 100:.1f}%)\n"
        )
        report_lines.append(f"- Tendencia: **{self._format_trend(food_trend)}**\n")

        report_lines.append("\n---\n\n")

        # Sección: Interpretación
        report_lines.append("## 💡 Interpretación\n\n")
        interpretation = self._generate_interpretation(
            summary, steps_trend, death_trend
        )
        report_lines.append(interpretation)

        # Generar gráficas si se solicita
        if include_plots:
            report_lines.append("\n---\n\n")
            report_lines.append("## 📈 Visualizaciones\n\n")

            plots_dir = os.path.join(os.path.dirname(output_path), "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Gráfica 1: Pasos por episodio
            steps_plot = self._plot_steps_over_time(plots_dir)
            if steps_plot:
                report_lines.append(f"### Pasos por Episodio\n\n")
                report_lines.append(f"![Pasos por episodio]({steps_plot})\n\n")

            # Gráfica 2: Distribución de pasos
            dist_plot = self._plot_steps_distribution(plots_dir)
            if dist_plot:
                report_lines.append(f"### Distribución de Pasos\n\n")
                report_lines.append(f"![Distribución de pasos]({dist_plot})\n\n")

            # Gráfica 3: Tasa de muerte inmediata
            death_plot = self._plot_death_rate_over_time(plots_dir)
            if death_plot:
                report_lines.append(f"### Tasa de Muerte Inmediata\n\n")
                report_lines.append(f"![Tasa de muerte inmediata]({death_plot})\n\n")

        # Escribir reporte
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(report_lines)

        print(f"✅ Reporte generado: {output_path}")
        return output_path

    def _format_trend(self, trend: str) -> str:
        """Formatea la tendencia para el reporte."""
        trend_map = {
            "improving": "Mejorando 📈",
            "worsening": "Empeorando 📉",
            "stable": "Estable ➡️",
            "insufficient_data": "Datos insuficientes ⏳",
            "unknown": "Desconocido ❓",
        }
        return trend_map.get(trend, trend)

    def _generate_interpretation(
        self, summary: Dict, steps_trend: str, death_trend: str
    ) -> str:
        """Genera interpretación automática de las métricas."""
        lines = []

        avg_steps = summary["avg_steps"]
        death_rate = summary["immediate_death_rate"]

        # Evaluación general
        if avg_steps > 50 and death_rate < 0.3:
            lines.append(
                "✅ **Progreso excelente**: El agente está sobreviviendo bien y evitando muertes inmediatas.\n\n"
            )
        elif avg_steps > 30 and death_rate < 0.5:
            lines.append(
                "🔄 **Progreso moderado**: El agente muestra señales de aprendizaje básico de supervivencia.\n\n"
            )
        elif avg_steps < 20 or death_rate > 0.7:
            lines.append(
                "⚠️ **Progreso limitado**: El agente aún tiene dificultades para sobrevivir más allá de los primeros pasos.\n\n"
            )
        else:
            lines.append(
                "🔍 **En desarrollo**: El agente está en fase temprana de aprendizaje.\n\n"
            )

        # Tendencias
        if steps_trend == "improving":
            lines.append(
                "- **Pasos promedio mejorando**: El agente está aprendiendo a sobrevivir más tiempo. ✅\n"
            )
        elif steps_trend == "worsening":
            lines.append(
                "- **Pasos promedio empeorando**: Posible sobreajuste o exploración excesiva. ⚠️\n"
            )

        if death_trend == "improving":
            lines.append(
                "- **Muerte inmediata reduciendo**: El agente está aprendiendo a evitar colisiones tempranas. ✅\n"
            )
        elif death_trend == "worsening":
            lines.append(
                "- **Muerte inmediata aumentando**: El agente puede estar explorando más o perdiendo estabilidad. ⚠️\n"
            )

        # Recomendaciones
        lines.append("\n**Recomendaciones:**\n\n")

        if death_rate > 0.5:
            lines.append(
                "1. Aumentar `SURVIVAL_REWARD` para reforzar la supervivencia básica\n"
            )
            lines.append(
                "2. Reducir temperatura de exploración para estabilizar el comportamiento\n"
            )

        if avg_steps < 30:
            lines.append(
                "1. Verificar que el agente está aprendiendo de las experiencias (revisar loss)\n"
            )
            lines.append(
                "2. Considerar aumentar el tamaño del batch de entrenamiento\n"
            )

        if steps_trend == "stable" and avg_steps < 50:
            lines.append(
                "1. El agente puede estar estancado - considerar ajustar learning rate\n"
            )
            lines.append(
                "2. Revisar si la memoria de replay tiene suficiente diversidad\n"
            )

        return "".join(lines)

    def _plot_steps_over_time(self, plots_dir: str) -> Optional[str]:
        """Genera gráfica de pasos por episodio con promedio móvil."""
        if not self.tracker or len(self.tracker.episodes) < 10:
            return None

        episodes = [ep.episode_num for ep in self.tracker.episodes]
        steps = [ep.steps for ep in self.tracker.episodes]

        # Calcular promedio móvil
        window = min(20, len(steps) // 5)
        moving_avg = np.convolve(steps, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, steps, alpha=0.3, label="Pasos por episodio", color="blue")
        plt.plot(
            episodes[window - 1 :],
            moving_avg,
            label=f"Promedio móvil ({window} episodios)",
            color="red",
            linewidth=2,
        )
        plt.axhline(
            y=self.tracker.early_death_threshold,
            color="orange",
            linestyle="--",
            label="Umbral de muerte inmediata",
        )
        plt.xlabel("Episodio")
        plt.ylabel("Pasos")
        plt.title("Pasos por Episodio - Evolución Temporal")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(plots_dir, "steps_over_time.png")
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_steps_distribution(self, plots_dir: str) -> Optional[str]:
        """Genera histograma de distribución de pasos."""
        if not self.tracker or len(self.tracker.episodes) < 10:
            return None

        steps = [ep.steps for ep in self.tracker.episodes]

        plt.figure(figsize=(10, 6))
        plt.hist(steps, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(
            x=np.mean(steps),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Promedio: {np.mean(steps):.1f}",
        )
        plt.axvline(
            x=self.tracker.early_death_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Umbral muerte inmediata: {self.tracker.early_death_threshold}",
        )
        plt.xlabel("Pasos")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de Pasos por Episodio")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        plot_path = os.path.join(plots_dir, "steps_distribution.png")
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        return plot_path

    def _plot_death_rate_over_time(self, plots_dir: str) -> Optional[str]:
        """Genera gráfica de tasa de muerte inmediata a lo largo del tiempo."""
        if not self.tracker or len(self.tracker.episodes) < 50:
            return None

        # Calcular tasa de muerte inmediata en ventanas deslizantes
        window = 50
        episode_nums = []
        death_rates = []

        for i in range(window, len(self.tracker.episodes) + 1):
            window_episodes = self.tracker.episodes[i - window : i]
            episode_nums.append(window_episodes[-1].episode_num)
            early_deaths = sum(1 for ep in window_episodes if ep.died_early)
            death_rates.append(early_deaths / window)

        plt.figure(figsize=(12, 6))
        plt.plot(episode_nums, death_rates, color="darkred", linewidth=2)
        plt.axhline(
            y=0.5, color="orange", linestyle="--", label="50% (umbral de alerta)"
        )
        plt.xlabel("Episodio")
        plt.ylabel("Tasa de Muerte Inmediata")
        plt.title(f"Tasa de Muerte Inmediata (ventana de {window} episodios)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        plot_path = os.path.join(plots_dir, "death_rate_over_time.png")
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        return plot_path

    def compare_periods(
        self, period1_start: int, period1_end: int, period2_start: int, period2_end: int
    ) -> Dict:
        """
        Compara métricas entre dos períodos.

        Args:
            period1_start: Inicio del período 1 (índice de episodio)
            period1_end: Fin del período 1
            period2_start: Inicio del período 2
            period2_end: Fin del período 2

        Returns:
            Diccionario con comparación de métricas
        """
        if not self.tracker:
            return {}

        period1 = self.tracker.episodes[period1_start:period1_end]
        period2 = self.tracker.episodes[period2_start:period2_end]

        comparison = {
            "period1": {
                "avg_steps": calculate_avg_steps(period1, len(period1)),
                "death_rate": calculate_immediate_death_rate(
                    period1, self.tracker.early_death_threshold, len(period1)
                ),
                "episodes": len(period1),
            },
            "period2": {
                "avg_steps": calculate_avg_steps(period2, len(period2)),
                "death_rate": calculate_immediate_death_rate(
                    period2, self.tracker.early_death_threshold, len(period2)
                ),
                "episodes": len(period2),
            },
        }

        # Calcular cambios
        comparison["changes"] = {
            "avg_steps_delta": comparison["period2"]["avg_steps"]
            - comparison["period1"]["avg_steps"],
            "death_rate_delta": comparison["period2"]["death_rate"]
            - comparison["period1"]["death_rate"],
        }

        return comparison


def quick_report_from_log(log_path: str, output_dir: str = "results/survival_reports"):
    """
    Función auxiliar para generar un reporte rápido desde un archivo de log.

    Args:
        log_path: Ruta al archivo de log
        output_dir: Directorio de salida para el reporte

    Returns:
        Ruta del reporte generado
    """
    reporter = SurvivalReporter()
    tracker = reporter.parse_log_file(log_path)
    reporter.tracker = tracker

    if len(tracker.episodes) == 0:
        print("⚠️ No se encontraron episodios en el log")
        return None

    # Generar nombre de archivo basado en timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"survival_report_{timestamp}.md")

    return reporter.generate_report(output_path, include_plots=True)
