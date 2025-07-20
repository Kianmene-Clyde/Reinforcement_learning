import numpy as np
from tqdm import tqdm

__all__ = ["sarsa", "q_learning", "expected_sarsa"]


def to_list(x):
    return list(x) if isinstance(x, (np.ndarray, tuple)) else x


def get_state(env):
    return env.state_id() if hasattr(env, "state_id") else env.state()


def extract_deterministic_policy(Q):
    return {s: int(np.argmax(Q[s])) for s in range(Q.shape[0])}


def epsilon_greedy_action(Q, s, available_actions, epsilon):
    available_actions = to_list(available_actions)
    if not available_actions:
        return None
    probs = np.ones(len(available_actions)) * (epsilon / len(available_actions))
    q_values = [Q[s, a] for a in available_actions]
    best_idx = int(np.argmax(q_values))
    probs[best_idx] += 1.0 - epsilon
    return np.random.choice(available_actions, p=probs)


def sarsa(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(episodes), desc="SARSA"):
        env.reset()
        s = get_state(env)
        old_score = env.score()
        available = to_list(env.available_actions())
        if not available:
            continue
        a = epsilon_greedy_action(Q, s, available, epsilon)
        if a is None:
            continue

        while not env.is_game_over():
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            available_prime = to_list(env.available_actions())
            if not available_prime:
                break
            a_prime = epsilon_greedy_action(Q, s_prime, available_prime, epsilon)
            if a_prime is None:
                break
            Q[s, a] += alpha * (r + gamma * Q[s_prime, a_prime] - Q[s, a])
            s, a = s_prime, a_prime

    policy = extract_deterministic_policy(Q)
    return {"policy": policy, "Q": Q}


def q_learning(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.3):
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(episodes), desc="Q-Learning"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            available = to_list(env.available_actions())
            if not available:
                break
            a = epsilon_greedy_action(Q, s, available, epsilon)
            if a is None:
                break

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            available_prime = to_list(env.available_actions())
            if available_prime:
                max_q = max(Q[s_prime, ap] for ap in available_prime)
            else:
                max_q = 0
            Q[s, a] += alpha * (r + gamma * max_q - Q[s, a])
            s = s_prime

    policy = extract_deterministic_policy(Q)
    return {"policy": policy, "Q": Q}


def expected_sarsa(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(episodes), desc="Expected SARSA"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            available = to_list(env.available_actions())
            if not available:
                break
            a = epsilon_greedy_action(Q, s, available, epsilon)
            if a is None:
                break

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            available_prime = to_list(env.available_actions())
            if available_prime:
                q_values = [Q[s_prime, ap] for ap in available_prime]
                probs = np.ones(len(available_prime)) * (epsilon / len(available_prime))
                best_idx = int(np.argmax(q_values))
                probs[best_idx] += 1.0 - epsilon
                expected_q = sum(p * q for p, q in zip(probs, q_values))
            else:
                expected_q = 0

            Q[s, a] += alpha * (r + gamma * expected_q - Q[s, a])
            s = s_prime

    policy = extract_deterministic_policy(Q)
    return {"policy": policy, "Q": Q}
