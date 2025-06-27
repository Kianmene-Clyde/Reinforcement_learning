import numpy as np
from tqdm import tqdm

__all__ = ["sarsa", "q_learning", "expected_sarsa"]


def get_num_actions(env):
    return env.num_actions() if callable(env.num_actions) else env.num_actions


def get_num_states(env):
    return env.num_states() if callable(env.num_states) else env.num_states


def get_state(env):
    return env.state_id() if hasattr(env, "state_id") else env.state()


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
        env.reset()
        s = get_state(env)
        pi = epsilon_greedy_policy(Q, epsilon)
        a = np.random.choice(np.arange(num_actions), p=pi[s])
        old_score = env.score()

        while not env.is_game_over():
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            pi = epsilon_greedy_policy(Q, epsilon)
            a_prime = np.random.choice(np.arange(num_actions), p=pi[s_prime])
            Q[s, a] += alpha * (r + gamma * Q[s_prime, a_prime] - Q[s, a])

            s, a = s_prime, a_prime

    return extract_deterministic_policy(Q), Q


def q_learning(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.3):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(episodes), desc="Q-Learning"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            pi = epsilon_greedy_policy(Q, epsilon)
            a = np.random.choice(np.arange(num_actions), p=pi[s])
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            target = r if env.is_game_over() else r + gamma * np.max(Q[s_prime])
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_prime

    return extract_deterministic_policy(Q), Q


def expected_sarsa(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(episodes), desc="Expected SARSA"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            pi = epsilon_greedy_policy(Q, epsilon)
            a = np.random.choice(np.arange(num_actions), p=pi[s])
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            pi_prime = epsilon_greedy_policy(Q, epsilon)
            expected_q = np.dot(pi_prime[s_prime], Q[s_prime])
            Q[s, a] += alpha * (r + gamma * expected_q - Q[s, a])
            s = s_prime

    return extract_deterministic_policy(Q), Q
