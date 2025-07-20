import numpy as np
from tqdm import tqdm
from collections import defaultdict

__all__ = ["dyna_q", "dyna_q_plus"]


def to_list(x):
    return list(x) if isinstance(x, (np.ndarray, tuple)) else x


def get_state(env):
    return env.state_id() if hasattr(env, "state_id") else env.state()


def dyna_q(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1, planning_steps=10):
    Q = defaultdict(lambda: defaultdict(float))  # Q-table
    model = defaultdict(dict)  # (s, a) -> (r, s')
    seen_state_action = set()  # pour les plans

    for _ in tqdm(range(episodes), desc="Dyna-Q"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            actions = to_list(env.available_actions())
            if not actions:
                break

            q_vals = [Q[s][a] for a in actions]
            pi = np.ones(len(actions)) * (epsilon / len(actions))
            best_idx = int(np.argmax(q_vals))
            pi[best_idx] += 1.0 - epsilon
            a = np.random.choice(actions, p=pi)

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            Q[s][a] += alpha * (r + gamma * max(Q[s_prime].values(), default=0) - Q[s][a])
            model[s][a] = (r, s_prime)
            seen_state_action.add((s, a))

            for _ in range(planning_steps):
                s_sim, a_sim = list(seen_state_action)[np.random.randint(len(seen_state_action))]
                r_sim, s_next_sim = model[s_sim][a_sim]
                Q[s_sim][a_sim] += alpha * (
                        r_sim + gamma * max(Q[s_next_sim].values(), default=0) - Q[s_sim][a_sim]
                )

            s = s_prime

    policy = {s: max(Q[s], key=Q[s].get) for s in Q if Q[s]}
    return {"policy": policy, "Q": Q}


def dyna_q_plus(env, episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1,
                planning_steps=10, kappa=1e-4):
    Q = defaultdict(lambda: defaultdict(float))  # Q-table
    model = defaultdict(dict)  # modèle pour le planning
    time_since = defaultdict(lambda: defaultdict(int))  # temps depuis la dernière visite
    seen_state_action = set()  # pour rééchantillonner

    for _ in tqdm(range(episodes), desc="Dyna-Q+"):
        env.reset()
        s = get_state(env)
        old_score = env.score()

        while not env.is_game_over():
            actions = to_list(env.available_actions())
            if not actions:
                break

            q_vals = [Q[s][a] for a in actions]
            pi = np.ones(len(actions)) * (epsilon / len(actions))
            best_idx = int(np.argmax(q_vals))
            pi[best_idx] += 1.0 - epsilon
            a = np.random.choice(actions, p=pi)

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = get_state(env)

            Q[s][a] += alpha * (r + gamma * max(Q[s_prime].values(), default=0) - Q[s][a])
            model[s][a] = (r, s_prime)
            time_since[s][a] = 0
            seen_state_action.add((s, a))

            for ss, aa in seen_state_action:
                if (ss, aa) != (s, a):
                    time_since[ss][aa] += 1

            for _ in range(planning_steps):
                s_sim, a_sim = list(seen_state_action)[np.random.randint(len(seen_state_action))]
                r_sim, s_next_sim = model[s_sim][a_sim]
                tau = time_since[s_sim][a_sim]
                bonus = kappa * np.sqrt(tau)
                target = r_sim + bonus + gamma * max(Q[s_next_sim].values(), default=0)
                Q[s_sim][a_sim] += alpha * (target - Q[s_sim][a_sim])

            s = s_prime

    policy = {s: max(Q[s], key=Q[s].get) for s in Q if Q[s]}
    return {"policy": policy, "Q": Q}
