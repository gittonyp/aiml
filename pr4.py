import heapq

def a_star(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    open_list = []
    heapq.heappush(open_list, (0, start))
    g_cost = {start: 0}
    came_from = {start: None}

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                new_cost = g_cost[current] + 1
                neighbor = (nx, ny)
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    f_cost = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_cost, neighbor))
                    came_from[neighbor] = current
    return None

maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)
path = a_star(maze, start, goal)

if path:
    print("Path found:", path)
    print("Total cost:", len(path)-1)
else:
    print("No path found")