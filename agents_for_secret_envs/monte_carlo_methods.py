import numpy as np
from tqdm import tqdm


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

    Q = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))
    pi = np.ones((num_states, num_actions)) * (1.0 / num_actions)
    all_actions = np.arange(num_actions)

    for _ in tqdm(range(episodes), desc="MC Control ε-soft"):
        env.reset()
        s = env.state_id()
        old_score = env.score()

        states, actions, rewards = [], [], []

        while not env.is_game_over():
            a = np.random.choice(all_actions, p=pi[s])
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = env.state_id()

        G = 0
        visited = set()

        for t in reversed(range(len(states))):
            s_t, a_t, r_tp1 = states[t], actions[t], rewards[t]
            G = gamma * G + r_tp1
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_count[s_t, a_t] += 1
                Q[s_t, a_t] += (G - Q[s_t, a_t]) / returns_count[s_t, a_t]

                best_a = np.argmax(Q[s_t])
                pi[s_t] = epsilon / num_actions
                pi[s_t, best_a] += 1.0 - epsilon

    return pi, Q


def monte_carlo_es(env, episodes=10000, gamma=0.99, max_steps=100):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)

    Q = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))
    pi = greedy_policy_from_q(Q)

    for _ in tqdm(range(episodes), desc="Monte Carlo Exploring Starts"):
        s0 = np.random.randint(0, num_states)
        a0 = np.random.randint(0, num_actions)

        try:
            env.reset_to(s0, a0)
        except Exception:
            continue

        episode = []
        old_score = env.score()
        s, a = s0, a0
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            episode.append((s, a, r))

            s = env.state_id()
            actions = env.available_actions()
            if actions:
                a = np.random.choice(actions)
            else:
                break

            step_count += 1

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, a_t, r_tp1 = episode[t]
            G = gamma * G + r_tp1
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_count[s_t, a_t] += 1
                Q[s_t, a_t] += (G - Q[s_t, a_t]) / returns_count[s_t, a_t]
                best_a = np.argmax(Q[s_t])
                pi[s_t] = np.eye(num_actions)[best_a]

    return pi, Q


def off_policy_mc_control(env, gamma=0.99, episodes=10000, max_steps=100):
    num_states = get_num_states(env)
    num_actions = get_num_actions(env)

    Q = np.zeros((num_states, num_actions))
    C = np.zeros((num_states, num_actions))
    pi = greedy_policy_from_q(Q)

    def behavior_policy(state):
        return np.ones(num_actions) / num_actions

    for _ in tqdm(range(episodes), desc="Off-Policy MC Control"):
        env.reset()
        s = env.state_id()
        old_score = env.score()
        episode = []
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            a = np.random.choice(np.arange(num_actions), p=behavior_policy(s))
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            episode.append((s, a, r))
            s = env.state_id()
            step_count += 1

        G = 0
        W = 1
        for t in reversed(range(len(episode))):
            s_t, a_t, r_tp1 = episode[t]
            G = gamma * G + r_tp1
            C[s_t, a_t] += W
            Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])
            pi[s_t] = np.eye(num_actions)[np.argmax(Q[s_t])]
            if pi[s_t][a_t] != 1.0:
                break
            W *= 1.0 / behavior_policy(s_t)[a_t]

    return pi, Q
