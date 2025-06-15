import numpy as np
from tqdm import tqdm

__all__ = ["sarsa", "q_learning", "expected_sarsa"]


def get_num_actions(env):
    return env.num_actions() if callable(env.num_actions) else env.num_actions


def get_num_states(env):
    return env.num_states() if callable(env.num_states) else env.num_states


def epsilon_greedy_policy(Q, epsilon):
    num_states, num_actions = Q.shape
    policy = np.ones((num_states, num_actions)) * (epsilon / num_actions)
    best_actions = np.argmax(Q, axis=1)
    policy[np.arange(num_states), best_actions] += 1.0 - epsilon
    return policy


def extract_deterministic_policy(Q):
    return {s: int(np.argmax(Q[s])) for s in range(Q.shape[0])}


def sarsa(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(episodes), desc="SARSA"):
        state = env.reset()
        s = env.get_state()
        pi = epsilon_greedy_policy(Q, epsilon)
        a = np.random.choice(num_actions, p=pi[s])
        old_score = env.score()

        while not env.is_game_over():
            try:
                next_state, _ = env.step(state, a)
            except TypeError:
                next_state, _ = env.step(a)

            r = env.score() - old_score
            old_score = env.score()
            s_prime = env.get_state()

            pi = epsilon_greedy_policy(Q, epsilon)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            a_prime = np.random.choice(num_actions, p=pi[s_prime])
            target = r + gamma * Q[s_prime, a_prime]
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_prime
            a = a_prime
            state = next_state

    final_policy = epsilon_greedy_policy(Q, epsilon)
    return final_policy, Q


def q_learning(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.3):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)
    Q = np.zeros((num_states, num_actions))
    visited_states = set()

    for _ in tqdm(range(episodes), desc="Q-Learning"):
        state = env.reset()
        s = env.get_state()
        pi = epsilon_greedy_policy(Q, epsilon)
        old_score = env.score()

        while not env.is_game_over():
            a = np.random.choice(num_actions, p=pi[s])

            try:
                next_state, reward = env.step(state, a)
            except TypeError:
                next_state, reward = env.step(a)

            r = env.score() - old_score
            old_score = env.score()
            s_prime = env.get_state()
            visited_states.add(s)

            max_q_next = np.max(Q[s_prime]) if not env.is_game_over() else 0
            target = r + gamma * max_q_next
            Q[s, a] += alpha * (target - Q[s, a])

            state = next_state
            s = s_prime

    deterministic_policy = extract_deterministic_policy(Q)
    return deterministic_policy, Q


def expected_sarsa(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(episodes), desc="Expected SARSA"):
        state = env.reset()
        s = env.get_state()
        old_score = env.score()

        while not env.is_game_over():
            pi = epsilon_greedy_policy(Q, epsilon)
            a = np.random.choice(num_actions, p=pi[s])

            try:
                next_state, _ = env.step(state, a)
            except TypeError:
                next_state, _ = env.step(a)

            r = env.score() - old_score
            old_score = env.score()
            s_prime = env.get_state()

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            pi_prime = epsilon_greedy_policy(Q, epsilon)
            expected_q = np.dot(pi_prime[s_prime], Q[s_prime])
            target = r + gamma * expected_q
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_prime
            state = next_state

    final_policy = epsilon_greedy_policy(Q, epsilon)
    return final_policy, Q
