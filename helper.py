import os
import sys
import pandas as pd
from IPython import display
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style
import numpy as np

def plot_training_progress(scores, mean_scores, save_plot=False, save_path="plots", filename="training_progress.png"):
    # Solo actualiza dinámicamente si se ejecuta en un entorno interactivo
    if "ipykernel" in sys.modules:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
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
            print(f"Plot saved to {full_path}.")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    plt.show(block=False)
    plt.pause(0.1)

def log_game_results(agent, score, record, game, avg_loss=None, rewards=None):
    timestamp = datetime.now()
    
    # Calculate efficiency metrics
    snake_length = len(game.snake) if hasattr(game, 'snake') else score + 1
    unique_positions = len(set((p.x, p.y) for p in game.snake)) if hasattr(game, 'snake') else 0
    efficiency_ratio = unique_positions / snake_length if snake_length > 0 else 0
    
    # Calculate reward statistics
    total_reward = sum(game.reward_history) if hasattr(game, 'reward_history') else 0
    avg_reward = total_reward / len(game.reward_history) if hasattr(game, 'reward_history') and len(game.reward_history) > 0 else 0
    max_reward = max(game.reward_history) if hasattr(game, 'reward_history') and len(game.reward_history) > 0 else 0
    min_reward = min(game.reward_history) if hasattr(game, 'reward_history') and len(game.reward_history) > 0 else 0
    
    # Calculate game duration and speed
    steps = game.steps if hasattr(game, 'steps') else 0
    steps_per_food = steps / score if score > 0 else steps
    
    # Calculate action distribution if available
    action_distribution = {}
    if hasattr(game, 'action_history') and len(game.action_history) > 0:
        actions = np.array(game.action_history)
        action_distribution = {
            'straight_pct': np.mean(actions == 0) * 100 if len(actions) > 0 else 0,
            'right_pct': np.mean(actions == 1) * 100 if len(actions) > 0 else 0,
            'left_pct': np.mean(actions == 2) * 100 if len(actions) > 0 else 0,
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
        except Exception as e:
            print(f"Could not get weight norms: {e}")
    
    # Create comprehensive game result entry
    game_result = {
        # Basic game info
        "game_number": agent.n_games,
        "score": score,
        "record": record,
        "timestamp": timestamp,
        
        # Performance metrics
        "snake_length": snake_length,
        "steps_taken": steps,
        "steps_per_food": steps_per_food,
        "efficiency_ratio": efficiency_ratio,
        
        # Reward metrics
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        
        # Training metrics
        "learning_rate": agent.trainer.lr if hasattr(agent, 'trainer') else None,
        "temperature": agent.temperature if hasattr(agent, 'temperature') else None,
        "avg_training_loss": avg_loss
    }
    
    # Add action distribution if available
    if action_distribution:
        game_result.update(action_distribution)
    
    # Add weight stats if available
    if weight_stats:
        game_result.update(weight_stats)
    
    # Add to game results and save periodically
    agent.game_results.append(game_result)
    
    if agent.n_games % 100 == 0:
        df_game_results = pd.DataFrame(agent.game_results)
        os.makedirs("results", exist_ok=True)
        
        # Save both detailed and summary CSVs
        df_game_results.to_csv("results/game_results_detailed.csv", index=False)
        
        # Create summary with rolling averages for key metrics
        if len(df_game_results) >= 10:  # Only if we have enough data
            summary_cols = ["game_number", "score", "steps_taken", "total_reward", "avg_reward", "efficiency_ratio"]
            available_cols = [col for col in summary_cols if col in df_game_results.columns]
            
            # Create rolling averages
            df_summary = df_game_results[available_cols].copy()
            for col in available_cols:
                if col != "game_number":
                    df_summary[f"{col}_avg10"] = df_game_results[col].rolling(10).mean()
                    df_summary[f"{col}_avg50"] = df_game_results[col].rolling(50).mean()
            
            df_summary.to_csv("results/game_results_summary.csv", index=False)
        
        print(f"Game results saved to CSV (game {agent.n_games})")

        # Generate analysis plots
        if len(df_game_results) >= 50:
            create_analysis_plots(df_game_results, "results/analysis_plots")
            
def create_analysis_plots(df, save_path="plots/analysis"):
    """Create detailed analysis plots from game results data."""
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Score progression with rolling averages
    plt.figure(figsize=(12, 6))
    plt.plot(df['game_number'], df['score'], color='blue', alpha=0.3, label='Score')
    plt.plot(df['game_number'], df['score'].rolling(20).mean(), color='red', label='20-game Avg')
    plt.plot(df['game_number'], df['score'].rolling(100).mean(), color='green', label='100-game Avg')
    plt.title('Score Progression')
    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f"{save_path}/score_progression.png")
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
    weight_cols = ['w1_norm', 'w2_norm', 'w3_norm']
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
    # Actualiza las listas de puntuaciones pero no muestra el gráfico durante el entrenamiento
    plot_scores.append(score)
    total_score += score
    mean_score = total_score / agent.n_games
    plot_mean_scores.append(mean_score)
    
    # Solo guardar el gráfico periódicamente, pero no mostrarlo
    save_plot = agent.n_games % 100 == 0
    if save_plot:
        plt.figure(figsize=(10, 6))  # Create a new figure
        plt.title("Training Progress")
        plt.xlabel("Number of Games")
        plt.ylabel("Score")
        plt.plot(plot_scores, label="Scores")
        plt.plot(plot_mean_scores, label="Mean Scores")
        plt.ylim(ymin=0)
        plt.legend()
        plt.text(len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1]))
        plt.text(len(plot_mean_scores) - 1, plot_mean_scores[-1], str(plot_mean_scores[-1]))
        
        # Save plot to file
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/training_progress_game_{agent.n_games}.png")
        plt.close()  # Close the figure to free memory
        
        print(f"Plot saved at game {agent.n_games}")
    
    return total_score

def save_checkpoint(agent, loss):
    # Keep whatever is in your current save_checkpoint function, but modify it to include record
    record_value = agent.record if hasattr(agent, 'record') else 0
    print(Fore.GREEN + f"Saving checkpoint with record: {record_value}" + Style.RESET_ALL)
    agent.model.save(
        "model_MARK_VII.pth",
        n_games=agent.n_games,
        optimizer=agent.trainer.optimizer.state_dict(),
        loss=loss,
        last_record_game=agent.last_record_game,
        record=record_value
    )

def print_game_info(reward, score, last_record_game, record):
    # Imprime información del juego
    print(f"Último récord obtenido en partida: {last_record_game}")
    print(Fore.CYAN + f"Reward: {reward:.4f}" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"Score: {score}" + Style.RESET_ALL)
    print(Fore.GREEN + f"Record:  {record}" + Style.RESET_ALL)
    print(Fore.RED + '-'*60 + Style.RESET_ALL)
    print('')
    print('')
    print(Fore.RED + '-'*60 + Style.RESET_ALL)
    

def print_weight_norms(agent):
    # Muestra las normas de los pesos para dar seguimiento al entrenamiento
    w1_norm = agent.model.linear1.weight.data.norm().item()
    w2_norm = agent.model.linear2.weight.data.norm().item()
    w3_norm = agent.model.linear3.weight.data.norm().item()
    print(Fore.CYAN + f"Weight norms: \nlinear1: {w1_norm:.4f} \nlinear2: {w2_norm:.4f} \nlinear3: {w3_norm:.4f}" + Style.RESET_ALL)
    print(Fore.RED + '-'*60 + Style.RESET_ALL)