import numpy as np
from utils import ACTIONS, getNextState, reward_fn, isValid

class QLearningPlanner:
    def __init__(self, mapArray, alpha=0.1, gamma=0.95, epsilon=1.0, episodes=5000, maxSteps=500):
        self.map = mapArray
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.maxSteps = maxSteps

    def _init_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.ones(len(ACTIONS)) * 1.0  # 乐观初始化，避免偏向左

    def choose_action(self, state):
        self._init_q(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(ACTIONS))
        return np.argmax(self.q_table[state])

    def train(self, start, goal):
        eps_min = 0.05
        eps_decay = 0.995
        self.reward_history = []

        for ep in range(self.episodes):
            state = start
            last_state = None
            total_reward = 0

            for _ in range(self.maxSteps):
                action_idx = self.choose_action(state)
                next_state = getNextState(state, ACTIONS[action_idx])
                r = reward_fn(next_state, goal, self.map, last_state)
                total_reward += r

                self._init_q(next_state)
                self.q_table[state][action_idx] += self.alpha * (r + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action_idx])

                if next_state == goal:
                    break
                if not isValid(next_state, self.map):
                    break

                last_state = state
                state = next_state

            self.epsilon = max(eps_min, self.epsilon * eps_decay)
            self.reward_history.append(total_reward)

    def extractPath(self, start, goal):
        state = start
        path = [state]
        visited = set()

        for _ in range(self.maxSteps):
            if state not in self.q_table:
                break
            visited.add(state)
            action_idx = np.argmax(self.q_table[state])
            next_state = getNextState(state, ACTIONS[action_idx])

            if not isValid(next_state, self.map) or next_state in visited:
                break
            path.append(next_state)
            if next_state == goal:
                return path
            state = next_state

        return path if path[-1] == goal else None


def crop_map_around_start_goal(mapArray, start, goal, margin=20):
    h, w = mapArray.shape
    min_x = max(0, min(start[0], goal[0]) - margin)
    max_x = min(h - 1, max(start[0], goal[0]) + margin)
    min_y = max(0, min(start[1], goal[1]) - margin)
    max_y = min(w - 1, max(start[1], goal[1]) + margin)

    cropped = mapArray[min_x:max_x+1, min_y:max_y+1]
    new_start = (start[0] - min_x, start[1] - min_y)
    new_goal = (goal[0] - min_x, goal[1] - min_y)

    return cropped, new_start, new_goal