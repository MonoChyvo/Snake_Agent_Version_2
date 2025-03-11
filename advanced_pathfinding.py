"""
advanced_pathfinding.py

Este módulo implementa estrategias avanzadas de búsqueda de caminos para el juego Snake,
inspiradas en la jugabilidad de humanos profesionales. El objetivo es lograr un juego perfecto
evitando quedar encerrado y maximizando la eficiencia en la recolección de comida.
"""

from heapq import heappush, heappop
from collections import deque

class AdvancedPathfinding:
    def __init__(self, game):
        self.game = game

    def find_optimal_path(self):
        """
        Decide dinámicamente entre buscar el camino más corto o el más largo
        basado en el estado actual del juego.
        """
        if self._should_seek_longest_path():
            return self.find_longest_path()
        else:
            return self.find_shortest_path()

    def find_shortest_path(self):
        """
        Encuentra el camino más corto hacia la comida utilizando el algoritmo A*.
        """
        start = self.game._grid_position(self.game.head)
        end = self.game._grid_position(self.game.food)
        grid_w, grid_h = self.game.grid_size

        open_list = []
        closed_set = set()

        start_node = self.Node(start)
        heappush(open_list, start_node)

        while open_list:
            current = heappop(open_list)

            if current.pos == end:
                path = []
                while current:
                    path.append(current.pos)
                    current = current.parent
                return path[::-1]  # Return reversed path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current.pos[0] + dx, current.pos[1] + dy)

                if self._is_position_valid(new_pos):
                    new_node = self.Node(new_pos, current)
                    new_node.g = current.g + 1
                    new_node.h = abs(new_pos[0] - end[0]) + abs(new_pos[1] - end[1])
                    new_node.f = new_node.g + new_node.h

                    if new_node.pos not in closed_set:
                        heappush(open_list, new_node)
                        closed_set.add(new_node.pos)

        return []  # No path found

    def find_longest_path(self):
        """
        Encuentra el camino más largo posible, evitando quedar encerrado.
        """
        start = self.game._grid_position(self.game.head)
        grid_w, grid_h = self.game.grid_size

        open_list = []
        closed_set = set()

        start_node = self.Node(start)
        heappush(open_list, start_node)

        longest_path = []
        while open_list:
            current = heappop(open_list)

            if current.g > len(longest_path):
                longest_path = []
                temp = current
                while temp:
                    longest_path.append(temp.pos)
                    temp = temp.parent
                longest_path = longest_path[::-1]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current.pos[0] + dx, current.pos[1] + dy)

                if self._is_position_valid(new_pos):
                    new_node = self.Node(new_pos, current)
                    new_node.g = current.g + 1
                    new_node.f = new_node.g  # Costo total es igual al costo acumulado

                    if new_node.pos not in closed_set:
                        heappush(open_list, new_node)
                        closed_set.add(new_node.pos)

        return longest_path

    def _should_seek_longest_path(self):
        """
        Decide si es mejor buscar el camino más largo basado en el espacio libre
        y el tamaño de la serpiente.
        """
        free_space = self._calculate_free_space(self.game._grid_position(self.game.head))
        return free_space < len(self.game.snake) * 2

    def _calculate_free_space(self, pos):
        """
        Calcula el espacio libre alrededor de una posición dada.
        """
        queue = deque([pos])
        visited = set()
        free_space = 0

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            free_space += 1

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current[0] + dx, current[1] + dy)
                if self._is_position_valid(new_pos):
                    queue.append(new_pos)

        return free_space

    def _is_position_valid(self, pos):
        """
        Verifica si una posición es válida (dentro del grid y no colisiona con la serpiente).
        """
        grid_w, grid_h = self.game.grid_size
        return (
            0 <= pos[0] < grid_w
            and 0 <= pos[1] < grid_h
            and not any(
                p.x // BLOCK_SIZE == pos[0] and p.y // BLOCK_SIZE == pos[1]
                for p in self.game.snake[1:]
            )
        )

    class Node:
        def __init__(self, pos, parent=None):
            self.pos = pos
            self.parent = parent
            self.g = 0
            self.h = 0
            self.f = 0

        def __lt__(self, other):
            return self.f < other.f