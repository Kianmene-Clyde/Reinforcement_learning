import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ----------- ENVIRONNEMENT GRIDWORLD 3x3 -----------
class MiniGridWorld:
    def __init__(self):
        self.grid_size = 3
        self.terminal_state = 8  # case en bas à droite (2,2)
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def num_states(self):
        return self.grid_size * self.grid_size

    def num_actions(self):
        return 4  # up, right, down, left

    def get_state(self):
        return self.state

    def is_game_over(self):
        return self.state == self.terminal_state

    def score(self):
        return 1 if self.state == self.terminal_state else 0

    def step(self, action):
        row, col = divmod(self.state, self.grid_size)

        if action == 0 and row > 0:  # up
            row -= 1
        elif action == 1 and col < self.grid_size - 1:  # right
            col += 1
        elif action == 2 and row < self.grid_size - 1:  # down
            row += 1
        elif action == 3 and col > 0:  # left
            col -= 1

        self.state = row * self.grid_size + col

# ----------- POLITIQUES DYNA-Q ET DYNA-Q+ -----------
def get_num_actions(env):
    return env.num_actions() if callable(env.num_actions) else env.num_actions

def get_num_states(env):
    return env.num_states() if callable(env.num_states) else env.num_states

def get_state(env):
    return env.get_state() if hasattr(env, "get_state") else env.state()

def epsilon_greedy_policy(Q, epsilon):
    num_states, num_actions = Q.shape
    pi = np.ones((num_states, num_actions)) * (epsilon / num_actions)
    best_actions = np.argmax(Q, axis=1)
    pi[np.arange(num_states), best_actions] += 1.0 - epsilon
    return pi

def dyna_q(env, episodes=200, gamma=0.99, alpha=0.1, epsilon=0.1, planning_steps=5):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)
    Q = np.zeros((num_states, num_actions))
    model = defaultdict(dict)
    seen_state_action = set()

    for _ in tqdm(range(episodes), desc="Dyna-Q"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            pi = epsilon_greedy_policy(Q, epsilon)
            a = np.random.choice(num_actions, p=pi[s])

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])
            model[s][a] = (r, s_prime)
            seen_state_action.add((s, a))

            for _ in range(planning_steps):
                s_sim, a_sim = list(seen_state_action)[np.random.randint(len(seen_state_action))]
                r_sim, s_next_sim = model[s_sim][a_sim]
                Q[s_sim, a_sim] += alpha * (r_sim + gamma * np.max(Q[s_next_sim]) - Q[s_sim, a_sim])

            s = s_prime

    return epsilon_greedy_policy(Q, epsilon), Q

def dyna_q_plus(env, episodes=200, gamma=0.99, alpha=0.1, epsilon=0.1, planning_steps=5, kappa=1e-2):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)
    Q = np.zeros((num_states, num_actions))
    model = defaultdict(dict)
    time_since = defaultdict(lambda: defaultdict(lambda: 0))
    seen_state_action = set()

    for _ in tqdm(range(episodes), desc="Dyna-Q+"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            pi = epsilon_greedy_policy(Q, epsilon)
            a = np.random.choice(num_actions, p=pi[s])

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])
            model[s][a] = (r, s_prime)
            seen_state_action.add((s, a))
            time_since[s][a] = 0

            for (ss, aa) in seen_state_action:
                if (ss, aa) != (s, a):
                    time_since[ss][aa] += 1

            for _ in range(planning_steps):
                s_sim, a_sim = list(seen_state_action)[np.random.randint(len(seen_state_action))]
                r_sim, s_next_sim = model[s_sim][a_sim]
                tau = time_since[s_sim][a_sim]
                bonus = kappa * np.sqrt(tau)
                target = r_sim + bonus + gamma * np.max(Q[s_next_sim])
                Q[s_sim, a_sim] += alpha * (target - Q[s_sim, a_sim])

            s = s_prime

    return epsilon_greedy_policy(Q, epsilon), Q

# ----------- TEST DES FONCTIONS -----------
env = MiniGridWorld()

print("\n▶️ Entraînement Dyna-Q...")
pi_dq, Q_dq = dyna_q(env)

print("\n▶️ Politique apprise (Dyna-Q) :")
print(np.argmax(pi_dq, axis=1).reshape((3, 3)))  # forme 3x3

print("\n▶️ Entraînement Dyna-Q+...")
pi_dq_plus, Q_dq_plus = dyna_q_plus(env)

print("\n▶️ Politique apprise (Dyna-Q+) :")
print(np.argmax(pi_dq_plus, axis=1).reshape((3, 3)))
