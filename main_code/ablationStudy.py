def reward_fn_no_shaping(state, goal, map_array, last_state=None):
    if not (0 <= state[0] < map_array.shape[0] and 0 <= state[1] < map_array.shape[1]) or map_array[state[0]][state[1]] == 1:
        return -50
    if state == goal:
        return 100
    return -1