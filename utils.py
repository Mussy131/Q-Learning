import torch
import numpy as np
import pickle
import warnings
import sys, types

# Add fake modules to avoid import errors
sys.modules['dataset'] = types.ModuleType('dataset')
sys.modules['dataset.map_sample'] = types.ModuleType('map_sample')

# Define a fake MapSample class to avoid import errors
class MapSample:
    def __init__(self):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state) 

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

sys.modules['dataset.map_sample'].__dict__['MapSample'] = MapSample

# Unleash torch warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# QLB8a (8 directions) action space
ACTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

def isValid(state, map_array):
    y, x = state
    return 0 <= x < map_array.shape[1] and 0 <= y < map_array.shape[0] and map_array[y][x] == 0

def getNextState(state, action):
    return (state[0] + action[0], state[1] + action[1])


# Reward function for the environment
# def reward_fn(state, goal, map_array, last_state=None):
#     if not isValid(state, map_array):
#         return -20
#     if state == goal:
#         return 100

#     reward = -1  # 每步基础代价

#     if last_state is not None:
#         last_dist = np.linalg.norm(np.array(last_state) - np.array(goal))
#         curr_dist = np.linalg.norm(np.array(state) - np.array(goal))
#         if curr_dist < last_dist:
#             reward += 2
#         else:
#             reward -= 2

#         dy = abs(state[0] - last_state[0])
#         dx = abs(state[1] - last_state[1])
#         if dx == 1 and dy == 1:
#             reward -= 0.5

#     return reward

def reward_fn(state, goal, map_array, last_state=None):
    if not isValid(state, map_array):
        return -50
    if state == goal:
        return 100

    reward = -1  

    if last_state is not None:
        last_dist = np.linalg.norm(np.array(last_state) - np.array(goal))
        curr_dist = np.linalg.norm(np.array(state) - np.array(goal))
        delta = last_dist - curr_dist

        reward += delta * 0.5  

    return reward

def loadPTSample(filepath):
    data = torch.load(filepath, map_location='cpu', weights_only=False)

    try:
        mapArray = data.map.numpy() # change map to numpy array
        start = tuple(int(v) for v in data.start)
        goal = tuple(int(v) for v in data.goal)
        return mapArray, start, goal
    except AttributeError as e:
        print(f"[ERROR] Failed to read from {filepath}: {e}")
        raise

