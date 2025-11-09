game_map = {
    'Start': ['Forest', 'Village'],
    'Forest': ['Cave', 'River'],
    'Village': ['Castle', 'Market'],
    'Cave': ['Treasure'],
    'River': [],
    'Castle': ['Dungeon'],
    'Market': [],
    'Dungeon': [],
    'Treasure': []
}

def find_all_paths_dfs(graph, node, goal, path, all_paths, order_visited):
    path = path + [node]
    
    if node not in order_visited:
        order_visited.append(node)

    if node == goal:
        all_paths.append(path)
    
    for neighbor in graph[node]:
        if neighbor not in path: # Avoid cycles in the current path
            find_all_paths_dfs(graph, neighbor, goal, path, all_paths, order_visited)

start = 'Start'
goal = 'Treasure'

all_paths = []
order = []
path = []

find_all_paths_dfs(game_map, start, goal, path, all_paths, order)

print("DFS Traversal Order (First exploration):")
print(" → ".join(order))

print("
All possible paths from Start to Treasure:")
for i, p in enumerate(all_paths, 1):
    print(f"Path {i}: {' → '.join(p)}")