from collections import deque

def water_jug(jug1_cap, jug2_cap, target):
    visited = set()
    q = deque([((0, 0), [(0, 0)])]) # Store (state, path)

    while q:
        (a, b), path = q.popleft()
        
        if (a, b) in visited:
            continue
        visited.add((a, b))

        if a == target or b == target:
            print("Goal reached!")
            print("Path of states (jug1, jug2):")
            for state in path:
                print(state)
            return path

        # 1. Fill jug1
        q.append(((jug1_cap, b), path + [(jug1_cap, b)]))
        # 2. Fill jug2
        q.append(((a, jug2_cap), path + [(a, jug2_cap)]))
        # 3. Empty jug1
        q.append(((0, b), path + [(0, b)]))
        # 4. Empty jug2
        q.append(((a, 0), path + [(a, 0)]))
        # 5. Pour jug1 to jug2
        pour_to_2 = min(a, jug2_cap - b)
        q.append(((a - pour_to_2, b + pour_to_2), path + [(a - pour_to_2, b + pour_to_2)]))
        # 6. Pour jug2 to jug1
        pour_to_1 = min(b, jug1_cap - a)
        q.append(((a + pour_to_1, b - pour_to_1), path + [(a + pour_to_1, b - pour_to_1)]))
        
    print("Goal cannot be reached.")
    return None

# Solve the 4-gallon jug, 3-gallon jug, get 2 gallons problem
water_jug(4, 3, 2)