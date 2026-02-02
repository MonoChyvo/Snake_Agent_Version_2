import pygame
import torch
import numpy as np
import builtins
from utils.logger import Logger
from utils.config import (
    MAX_EPOCHS,
    BATCH_SIZE,
    LR,
    MIN_TEMPERATURE,
    DECAY_RATE,
    EXPLORATION_FREQUENCY,
    EXPLORATION_TEMP,
    EXPLORATION_DURATION,
    SURVIVAL_METRICS,
)
from src.agent import Agent
from src.game import SnakeGameAI
from utils.helper import log_game_results, update_game_summary, print_weight_norms
from utils.plotting import update_plots
from utils.evaluation import evaluate_agent, print_evaluation_results
from src.core.events import event_system
from utils.advanced_pathfinding import AdvancedPathfinding
from utils.survival_metrics import SurvivalMetricsTracker


def train(max_games: int) -> None:
    # Inicializar pygame si no está inicializado
    if not pygame.get_init():
        pygame.init()

    Logger.print_training_start()

    agent = Agent()
    globals()["agent"] = agent
    setattr(builtins, "agent", agent)

    game = SnakeGameAI(agent=agent)
    agent.game = game

    record = agent.record
    total_score = 0
    plot_mean_scores = []
    plot_scores = []

    # Inicializar tracker de métricas de supervivencia
    survival_tracker = SurvivalMetricsTracker(
        window_size=SURVIVAL_METRICS["survival_window"],
        early_death_threshold=SURVIVAL_METRICS["early_death_threshold"],
    )

    # Sincronización inicial del estado compartido
    agent.exploration_strategy._sync_shared_state(agent.pathfinding_enabled)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(game, state_old)
        reward, done, score = game.play_step(final_move, agent.n_games, record)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        agent.update_target_network()

        if done:
            episode_reward = sum(game.reward_history)
            avg_reward = (
                episode_reward / len(game.reward_history) if game.reward_history else 0
            )

            if not hasattr(agent, "recent_scores"):
                agent.recent_scores = []
            agent.recent_scores.append(score)
            if len(agent.recent_scores) > 50:
                agent.recent_scores.pop(0)

            loss = agent.train_long_memory()
            log_game_results(agent, score, record, game, avg_loss=loss)

            # Registrar episodio en tracker de supervivencia
            steps_taken = game.steps
            survival_tracker.record_episode(
                episode_num=agent.n_games + 1,
                steps=steps_taken,
                score=score,
                cause_of_death="collision",
            )

            # Actualizar exploración delegando a la estrategia
            temp, phase, left = agent.exploration_strategy.update(
                agent.n_games, agent.pathfinding_enabled
            )
            agent.temperature = temp
            agent.exploration_phase = phase
            agent.exploration_games_left = left

            # Ajustar tasa de aprendizaje dinámicamente
            current_lr = agent.adjust_learning_rate()

            agent.n_games += 1
            agent.monitor_memory()
            game.reset()

            # Guardar el mejor record
            if score > record:
                record = score
                agent.record = score
                agent.last_record_game = agent.n_games
                game.record = score  # Sincronizar inmediatamente con el juego
                Logger.print_success(
                    f"¡Nuevo récord en el juego {agent.last_record_game}!"
                )

                # Sincronizar estado al haber nuevo record
                update_game_summary(game=game, agent=agent, force_update=True)
                if hasattr(game, "stats_manager"):
                    game.stats_manager.update()

            # Guardar checkpoint
            agent.checkpoint_manager.save(
                agent.n_games,
                agent.last_record_game,
                agent.record,
                agent.pathfinding_enabled,
                agent.temperature,
            )

            # Actualizar gráficas
            total_score = update_plots(
                agent, score, total_score, plot_scores, plot_mean_scores
            )

            Logger.print_game_header(agent.n_games)

            # Métricas de eficiencia
            eff_ratio = (
                len(set((p.x, p.y) for p in game.snake)) / len(game.snake)
                if len(game.snake) > 0
                else 0
            )
            steps_pf = steps_taken / score if score > 0 else steps_taken
            agent.efficiency_ratio = eff_ratio
            agent.steps_per_food = steps_pf

            game_stats = {
                "score": score,
                "steps": steps_taken,
                "record": record,
                "total_reward": episode_reward,
                "avg_reward": avg_reward,
                "loss": loss,
                "temperature": agent.temperature,
                "last_record_game": agent.last_record_game,
                "learning_rate": current_lr,
                "efficiency_ratio": eff_ratio,
                "steps_per_food": steps_pf,
            }
            Logger.print_game_summary(game_stats)

            # Mostrar métricas de supervivencia cada N episodios
            if agent.n_games % SURVIVAL_METRICS["report_frequency"] == 0:
                survival_summary = survival_tracker.get_summary()
                Logger.print_survival_metrics(survival_summary)

            # Generar reporte completo cada 100 episodios
            if agent.n_games % 100 == 0 and agent.n_games > 0:
                from utils.survival_reporter import SurvivalReporter
                import os

                reporter = SurvivalReporter(survival_tracker)
                report_dir = "results/survival_reports"
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(
                    report_dir, f"survival_report_ep{agent.n_games}.md"
                )
                reporter.generate_report(report_path, include_plots=True)
                Logger.print_success(
                    f"Reporte de supervivencia generado: {report_path}"
                )

            # Gestión automática de pathfinding desactivada por requerimiento
            # if agent.exploration_phase:
            #     agent.set_pathfinding(False)
            #     Logger.print_exploration_status(True, agent.exploration_games_left)
            # else:
            #     agent.set_pathfinding(True)
            #     Logger.print_exploration_status(False)

            # Forzar visibilidad de pathfinding desactivado en el log
            agent.set_pathfinding(False)
            Logger.print_exploration_status(False)

            # Actualizar normas de pesos
            norms = print_weight_norms(agent)
            if norms:
                event_system.notify("weight_norms_updated", norms)

            # Sincronización final del loop
            update_game_summary(game=game, agent=agent, force_update=True)

            # Evaluación periódica
            if agent.n_games % 100 == 0:
                print_evaluation_results(
                    evaluate_agent(agent, num_episodes=5), is_periodic=True
                )

            if agent.n_games >= max_games:
                Logger.print_training_end()
                print_evaluation_results(
                    evaluate_agent(agent, num_episodes=10), is_periodic=False
                )
                break


if __name__ == "__main__":
    train(MAX_EPOCHS)
