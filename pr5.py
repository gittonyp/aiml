import heapq

def heuristic(state, goal):
    return sum(abs(s//3 - g//3) + abs(s%3 - g%3)
               for s, g in ((state.index(i), goal.index(i)) for i in range(1,9)))

def get_neighbors(state):
    neighbors = []
    i = state.index(0)
    x, y = divmod(i, 3)
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    for dx, dy in moves:
        nx, ny = x+dx, y+dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            ni = nx*3+ny
            new_state = list(state)
            new_state[i], new_state[ni] = new_state[ni], new_state[i]
            neighbors.append(tuple(new_state))
    return neighbors

def a_star_8_puzzle(start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start))
    came_from = {start: None}
    g_cost = {start: 0}
    
    closed_set = set()

    while open_list:
        f, cost, current = heapq.heappop(open_list)
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]
            
        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
                
            new_cost = g_cost[current] + 1
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_cost, new_cost, neighbor))
                came_from[neighbor] = current
    return None

start = (1,2,3,4,0,6,7,5,8)
goal = (1,2,3,4,5,6,7,8,0)
path = a_star_8_puzzle(start, goal)

if path:
    print("Solution found:")
    for step in path:
        print(step[:3])
        print(step[3:6])
        print(step[6:])
        print("-" * 5)
    print("Total Steps:", len(path)-1)
else:
    print("No solution found.")