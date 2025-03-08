from agent import Agent
from game import SnakeGameAI

agent = Agent()
game = SnakeGameAI()

print("Recent Scores:", agent.recent_scores)
print("Record:", agent.record)
print(game.visit_map)
