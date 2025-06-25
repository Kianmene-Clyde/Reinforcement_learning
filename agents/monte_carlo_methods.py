import numpy as np
from tqdm import tqdm

__all__ = [
    "on_policy_first_visit_mc_control",
    "monte_carlo_es",
    "off_policy_mc_control",
]


def get_num_states(env):
    return env.num_states() if callable(env.num_states) else env.num_states


def get_num_actions(env):
    return env.num_actions() if callable(env.num_actions) else env.num_actions


def greedy_policy_from_q(Q):
    num_states, num_actions = Q.shape
    pi = np.zeros((num_states, num_actions))
    for s in range(num_states):
        best_a = np.argmax(Q[s])
        pi[s, best_a] = 1.0
    return pi


def on_policy_first_visit_mc_control(env, episodes=10000, gamma=0.99, epsilon=0.1):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)

    Q = np.random.random((num_states, num_actions))
    Returns_count = np.zeros((num_states, num_actions))
    pi = np.ones((num_states, num_actions)) * (1.0 / num_actions)
    all_actions = np.arange(num_actions)

    for _ in tqdm(range(episodes), desc="MC Control ε-soft"):
        env.reset()
        states, actions, rewards = [], [], []

        while not env.is_game_over():
            s = env.get_state()
            a = np.random.choice(all_actions, p=pi[s])
            old_score = env.score()
            env.step(a)
            r = env.score() - old_score

            states.append(s)
            actions.append(a)
            rewards.append(r)

        terminal_state = env.get_state()
        Q[terminal_state, :] = 0.0

        G = 0
        visited = set()

        for t in reversed(range(len(states))):
            s_t, a_t, r_tp1 = states[t], actions[t], rewards[t]
            G = gamma * G + r_tp1

            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                Returns_count[s_t, a_t] += 1
                Q[s_t, a_t] += (G - Q[s_t, a_t]) / Returns_count[s_t, a_t]

                best_a = np.argmax(Q[s_t])
                pi[s_t] = epsilon / num_actions
                pi[s_t, best_a] = 1.0 - epsilon + (epsilon / num_actions)

    return pi, Q


def monte_carlo_es(env, episodes=10000, gamma=0.99, max_steps=100):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)

    try:
        Q = np.random.random((num_states, num_actions))
        Returns_sum = np.zeros((num_states, num_actions))
        Returns_count = np.zeros((num_states, num_actions))
        pi = greedy_policy_from_q(Q)

        for _ in tqdm(range(episodes), desc="Monte Carlo Exploring Starts"):
            s0 = np.random.randint(0, num_states)
            a0 = np.random.randint(0, num_actions)
            env.reset_to(s0, a0)

            episode = []
            visited_pairs = set()
            old_score = env.score()

            step_count = 0
            while not env.is_game_over() and step_count < max_steps:
                s = env.get_state()
                a = np.argmax(pi[s])
                env.step(a)
                r = env.score() - old_score
                old_score = env.score()
                episode.append((s, a, r))
                step_count += 1

            if step_count >= max_steps:
                print("⚠️ Avertissement : max_steps atteint dans monte_carlo_es")

            G = 0
            for t in reversed(range(len(episode))):
                s_t, a_t, r_tp1 = episode[t]
                G = gamma * G + r_tp1

                if (s_t, a_t) not in visited_pairs:
                    visited_pairs.add((s_t, a_t))
                    Returns_count[s_t][a_t] += 1
                    Returns_sum[s_t][a_t] += (G - Returns_sum[s_t][a_t]) / Returns_count[s_t][a_t]
                    Q[s_t][a_t] = Returns_sum[s_t][a_t]
                    pi[s_t] = np.eye(num_actions)[np.argmax(Q[s_t])]

        return pi, Q

    except MemoryError:
        print("❌ MemoryError : trop d'états pour Monte Carlo ES.")
        fallback_q = Q if 'Q' in locals() else np.zeros((num_states, num_actions))
        return np.zeros((num_states, num_actions)), fallback_q


def off_policy_mc_control(env, episodes=10000, gamma=0.99):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)

    Q = np.random.random((num_states, num_actions))
    C = np.zeros((num_states, num_actions))
    pi = greedy_policy_from_q(Q)
    b = np.ones((num_states, num_actions)) / num_actions

    for _ in tqdm(range(episodes), desc="Off-Policy MC Control"):
        env.reset()
        states, actions, rewards = [], [], []
        old_score = env.score()

        while not env.is_game_over():
            s = env.get_state()
            a = np.random.choice(np.arange(num_actions), p=b[s])
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()

            states.append(s)
            actions.append(a)
            rewards.append(r)

        G = 0
        W = 1

        for t in reversed(range(len(states))):
            s_t, a_t, r_tp1 = states[t], actions[t], rewards[t]
            G = gamma * G + r_tp1

            C[s_t, a_t] += W
            Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])
            pi[s_t] = np.eye(num_actions)[np.argmax(Q[s_t])]

            if a_t != np.argmax(pi[s_t]):
                break
            W = W / b[s_t, a_t]

    return pi, Q
