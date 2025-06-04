import numpy as np
from tqdm import tqdm
from collections import defaultdict

__all__ = ["dyna_q", "dyna_q_plus", "epsilon_greedy_policy"]


def epsilon_greedy_policy(Q, epsilon):
    """
    Génère une politique ε-greedy à partir de Q.
    """
    num_states, num_actions = Q.shape
    pi = np.ones((num_states, num_actions)) * (epsilon / num_actions)
    best_actions = np.argmax(Q, axis=1)
    pi[np.arange(num_states), best_actions] += 1.0 - epsilon
    return pi


def dyna_q(env, num_episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1, planning_steps=10):
    """
    Implémentation de Dyna-Q tabulaire (modèle déterministe)

    Args:
        env: environnement avec .reset(), .state(), .step(a), .score(), .is_game_over(), .num_states(), .num_actions()
        num_episodes: nombre d’épisodes à exécuter
        gamma: facteur de discount
        alpha: taux d’apprentissage
        epsilon: ε-greedy pour la politique
        planning_steps: nombre de simulations à faire à chaque pas réel

    Returns:
        pi: politique ε-greedy finale
        Q: tableau Q(s, a)
    """
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    # Modèle déterministe : model[s][a] = (r, s')
    model = defaultdict(dict)

    # Historique des états/actions observés pour échantillonnage
    seen_state_action = set()

    for _ in tqdm(range(num_episodes), desc="Dyna-Q"):
        env.reset()
        s = env.state()
        old_score = env.score()

        while not env.is_game_over():
            pi = epsilon_greedy_policy(Q, epsilon)
            a = np.random.choice(num_actions, p=pi[s])

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = env.state()

            # (d) Mise à jour de Q-table (vraie interaction)
            best_next = np.max(Q[s_prime])
            Q[s, a] += alpha * (r + gamma * best_next - Q[s, a])

            # (e) Mise à jour du modèle
            model[s][a] = (r, s_prime)
            seen_state_action.add((s, a))

            # (f) Simulations (planification)
            for _ in range(planning_steps):
                s_sim, a_sim = list(seen_state_action)[
                    np.random.randint(len(seen_state_action))
                ]
                r_sim, s_next_sim = model[s_sim][a_sim]
                best_q_next = np.max(Q[s_next_sim])
                Q[s_sim, a_sim] += alpha * (r_sim + gamma * best_q_next - Q[s_sim, a_sim])

            s = s_prime

    pi_finale = epsilon_greedy_policy(Q, epsilon)
    return pi_finale, Q


def dyna_q_plus(env, num_episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1,
                planning_steps=10, kappa=1e-4):
    """
    Implémentation de Dyna-Q+ avec bonus d’exploration (κ√τ)

    Args:
        env: environnement avec .reset(), .state(), .step(a), .score(), .is_game_over(), .num_states(), .num_actions()
        num_episodes: nombre d’épisodes à exécuter
        gamma: facteur de discount
        alpha: taux d’apprentissage
        epsilon: ε-greedy pour la politique
        planning_steps: nombre de mises à jour simulées à chaque pas réel
        kappa: coefficient du bonus d’exploration

    Returns:
        pi: politique ε-greedy finale
        Q: tableau Q(s, a)
    """
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    # Modèle : model[s][a] = (r, s')
    model = defaultdict(dict)

    # Temps depuis la dernière visite de (s,a)
    time_since = defaultdict(lambda: defaultdict(lambda: 0))

    # Historique des transitions observées
    seen_state_action = set()

    total_steps = 0

    for _ in tqdm(range(num_episodes), desc="Dyna-Q+"):
        env.reset()
        s = env.state()
        old_score = env.score()

        while not env.is_game_over():
            total_steps += 1

            pi = epsilon_greedy_policy(Q, epsilon)
            a = np.random.choice(num_actions, p=pi[s])

            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = env.state()

            # Mise à jour réelle de Q
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])

            # Mise à jour du modèle et horloge
            model[s][a] = (r, s_prime)
            seen_state_action.add((s, a))
            time_since[s][a] = 0  # Reset

            # Incrémenter le "temps non visité" de tous les autres (s,a)
            for (ss, aa) in seen_state_action:
                if (ss, aa) != (s, a):
                    time_since[ss][aa] += 1

            # Planification avec bonus
            for _ in range(planning_steps):
                s_sim, a_sim = list(seen_state_action)[
                    np.random.randint(len(seen_state_action))
                ]
                r_sim, s_next_sim = model[s_sim][a_sim]
                tau = time_since[s_sim][a_sim]
                bonus = kappa * np.sqrt(tau)
                target = r_sim + bonus + gamma * np.max(Q[s_next_sim])
                Q[s_sim, a_sim] += alpha * (target - Q[s_sim, a_sim])

            s = s_prime

    pi_finale = epsilon_greedy_policy(Q, epsilon)
    return pi_finale, Q
