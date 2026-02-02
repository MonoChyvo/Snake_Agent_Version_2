"""
Script de análisis de supervivencia standalone.

Extrae métricas de supervivencia de archivos de log existentes y genera
reportes completos sin necesidad de reentrenar el agente.

Uso:
    python scripts/analyze_survival.py --log logs/snake_dqn_20260201.log --output results/survival_reports/

Argumentos:
    --log: Ruta al archivo de log a analizar
    --output: Directorio de salida para el reporte (opcional)
    --window: Tamaño de ventana para promedios móviles (opcional, default: 100)
    --threshold: Umbral de muerte inmediata en pasos (opcional, default: 10)

Versión: 1.0.0
"""

import argparse
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.survival_reporter import SurvivalReporter, quick_report_from_log
from utils.survival_metrics import SurvivalMetricsTracker


def main():
    parser = argparse.ArgumentParser(
        description="Analiza logs de entrenamiento y genera reportes de supervivencia"
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Ruta al archivo de log a analizar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/survival_reports",
        help="Directorio de salida para el reporte",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Tamaño de ventana para promedios móviles",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Umbral de muerte inmediata en pasos",
    )

    args = parser.parse_args()

    # Validar que el archivo de log existe
    if not os.path.exists(args.log):
        print(f"❌ Error: Archivo de log no encontrado: {args.log}")
        sys.exit(1)

    print(f"📊 Analizando log: {args.log}")
    print(f"📁 Directorio de salida: {args.output}")
    print(f"🔧 Ventana: {args.window} episodios")
    print(f"⚠️ Umbral muerte inmediata: {args.threshold} pasos")
    print("\n" + "=" * 60 + "\n")

    # Crear reporter y parsear log
    reporter = SurvivalReporter()
    tracker = reporter.parse_log_file(args.log)
    reporter.tracker = tracker

    if len(tracker.episodes) == 0:
        print("⚠️ No se encontraron episodios en el log")
        print("Verifica que el formato del log sea correcto.")
        sys.exit(1)

    print(f"✅ Episodios encontrados: {len(tracker.episodes)}")

    # Mostrar resumen rápido en consola
    summary = tracker.get_summary()
    print("\n📊 Resumen de Métricas de Supervivencia:\n")
    print(
        f"  Pasos promedio (últimos {summary['window_size']}): {summary['avg_steps']:.1f}"
    )
    print(f"  Muerte inmediata: {summary['immediate_death_rate'] * 100:.1f}%")
    print(f"  Episodios supervivencia (≥50 pasos): {summary['survival_episodes']}")

    food_stats = summary["food_stats"]
    print(f"  Comidas promedio: {food_stats['avg_food']:.2f}")
    print(
        f"  Episodios con comida: {food_stats['episodes_with_food']} ({food_stats['food_rate'] * 100:.1f}%)"
    )

    # Detectar tendencias
    steps_trend = tracker.get_trend("steps")
    death_trend = tracker.get_trend("death_rate")
    food_trend = tracker.get_trend("food")

    print(f"\n📈 Tendencias:")
    print(f"  Pasos: {steps_trend}")
    print(f"  Muerte inmediata: {death_trend}")
    print(f"  Comidas: {food_trend}")

    # Generar reporte completo
    print("\n" + "=" * 60)
    print("📝 Generando reporte completo...\n")

    os.makedirs(args.output, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.output, f"survival_report_{timestamp}.md")

    reporter.generate_report(report_path, include_plots=True)

    print(f"\n✅ Análisis completado")
    print(f"📄 Reporte: {report_path}")
    print(f"📊 Gráficas: {os.path.join(args.output, 'plots')}")


if __name__ == "__main__":
    main()
