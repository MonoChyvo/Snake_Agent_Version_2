from agent import Agent
from game import SnakeGameAI

agent = Agent()
game = SnakeGameAI()

print("Recent Scores:", agent.recent_scores)
print("Record:", agent.record)
print(game.visit_map)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar CSV; se ignoran líneas de comentarios iniciadas con '//' 
df = pd.read_csv(r"c:\Users\chivo\OneDrive\Escritorio\Ciencia De Datos\Reinforcement Learning\snake_0020_DQN\results\df_metrics.csv", comment='/')

# Mostrar estadísticos básicos de algunas columnas clave
cols = ['loss', 'w1_norm', 'w2_norm', 'w3_norm', 'gradient_norm', 'weight_update_magnitude']
print("Resumen estadístico:")
print(df[cols].describe())

# Calcular matriz de correlación entre las métricas seleccionadas
corr_matrix = df[cols].corr()
print("\nMatriz de correlación:")
print(corr_matrix)

# Graficar la relación entre la pérdida y las normas de pesos
plt.figure(figsize=(12, 8))
for col in ['w1_norm', 'w2_norm', 'w3_norm']:
    sns.scatterplot(data=df, x=col, y='loss', label=col)
plt.xlabel("Norma de Pesos")
plt.ylabel("Loss")
plt.title("Relación entre Loss y Normas de Pesos")
plt.legend()
plt.show()

# Graficar la relación entre el gradiente y la pérdida
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='gradient_norm', y='loss')
plt.xlabel("Norma del Gradiente")
plt.ylabel("Loss")
plt.title("Relación entre Loss y Norma del Gradiente")
plt.show()