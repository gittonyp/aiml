import heapq
import math

def a_star(start, goal, graph, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    while open_list:
        current_f, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        for neighbor, cost in graph[current].items():
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic[neighbor]
                heapq.heappush(open_list, (f_score, neighbor))
    return None, float('inf')

graph = {
    'A': {'B': 4, 'C': 3},
    'B': {'A': 4, 'D': 2, 'E': 5},
    'C': {'A': 3, 'D': 6},
    'D': {'B': 2, 'C': 6, 'E': 1, 'F': 7},
    'E': {'B': 5, 'D': 1, 'F': 3},
    'F': {'D': 7, 'E': 3}
}

heuristic = {
    'A': 10,
    'B': 8,
    'C': 9,
    'D': 5,
    'E': 3,
    'F': 0
}

start = 'A'
goal = 'F'
path, cost = a_star(start, goal, graph, heuristic)

print("Shortest Path found by A*:", path)
print("Total Cost:", cost)