import heapq
import numpy as np

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(map_array, start, goal):
    h, w = map_array.shape
    visited = set()
    heap = [(0 + heuristic(start, goal), 0, start, [])]  # (f, g, state, path)

    while heap:
        f, g, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        new_path = path + [current]
        if current == goal:
            return new_path

        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ny, nx = current[0] + dy, current[1] + dx
            neighbor = (ny, nx)
            if 0 <= ny < h and 0 <= nx < w and map_array[ny][nx] == 0 and neighbor not in visited:
                cost = g + heuristic(current, neighbor)
                heapq.heappush(heap, (cost + heuristic(neighbor, goal), cost, neighbor, new_path))
    return None

def astar_with_q_heuristic(map_array, start, goal, q_table, weight=1.0):
    h, w = map_array.shape
    visited = set()
    heap = [(0 + heuristic(start, goal), 0, start, [])]

    def max_q(state):
        return np.max(q_table.get(state, np.zeros(8)))

    while heap:
        f, g, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        new_path = path + [current]
        if current == goal:
            return new_path

        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ny, nx = current[0] + dy, current[1] + dx
            neighbor = (ny, nx)
            if 0 <= ny < h and 0 <= nx < w and map_array[ny][nx] == 0 and neighbor not in visited:
                cost = g + heuristic(current, neighbor)
                h_score = heuristic(neighbor, goal) - weight * max_q(neighbor)
                heapq.heappush(heap, (cost + h_score, cost, neighbor, new_path))
    return None
