from collections import deque

def bfs_shortest_path(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    visited = [[False]*cols for _ in range(rows)]
    parent = [[None]*cols for _ in range(rows)]

    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    queue = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        x, y = queue.popleft()

        if (x, y) == goal:
            path = []
            pos = goal
            while pos is not None:
                path.append(pos)
                pos = parent[pos[0]][pos[1]]
            path.reverse()
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and maze[nx][ny] == 0:
                visited[nx][ny] = True
                parent[nx][ny] = (x, y)
                queue.append((nx, ny))

    return None  # No path found

maze = [
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0]
]

start = (0, 0)   # Starting cell
goal = (4, 4)    # Goal cell

path = bfs_shortest_path(maze, start, goal)

if path:
    print("Shortest path found using BFS:")
    print(path)
    print("Total steps:", len(path)-1)
else:
    print("No path found!")