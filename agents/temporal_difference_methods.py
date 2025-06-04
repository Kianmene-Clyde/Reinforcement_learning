import numpy as np
from tqdm import tqdm

__all__ = ["sarsa", "q_learning", "expected_sarsa", "epsilon_greedy_policy"]


def epsilon_greedy_policy(Q, epsilon):
    """
    Génère une politique ε-greedy à partir de Q.
    """
    num_states, num_actions = Q.shape
    pi = np.ones((num_states, num_actions)) * (epsilon / num_actions)
    best_actions = np.argmax(Q, axis=1)
    pi[np.arange(num_states), best_actions] += 1.0 - epsilon
    return pi


def sarsa(env, num_episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    SARSA (on-policy TD control) pour approximer Q ≈ q*.

    Args:
        env: environnement avec reset(), step(a), state(), score(), is_game_over()
        num_episodes: nombre d'épisodes à exécuter
        gamma: facteur de discount
        alpha: taux d’apprentissage
        epsilon: taux d’exploration (ε-greedy)

    Returns:
        pi: politique ε-greedy finale
        Q: tableau des valeurs Q(s, a)
    """
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(num_episodes), desc="SARSA"):
        env.reset()
        s = env.state()
        pi = epsilon_greedy_policy(Q, epsilon)
        a = np.random.choice(num_actions, p=pi[s])
        old_score = env.score()

        while not env.is_game_over():
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = env.state()

            pi = epsilon_greedy_policy(Q, epsilon)
            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            a_prime = np.random.choice(num_actions, p=pi[s_prime])
            target = r + gamma * Q[s_prime, a_prime]
            Q[s, a] += alpha * (target - Q[s, a])

            s, a = s_prime, a_prime

    final_policy = epsilon_greedy_policy(Q, epsilon)
    return final_policy, Q


def q_learning(env, num_episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Q-learning (off-policy TD control) pour estimer π ≈ π*

    Args:
        env: environnement avec .reset(), .state(), .step(a), .score(), .is_game_over(), .num_states(), .num_actions()
        num_episodes: nombre d’épisodes à exécuter
        gamma: facteur de discount
        alpha: taux d’apprentissage
        epsilon: taux d’exploration

    Returns:
        pi: politique finale ε-greedy
        Q: tableau Q(s, a)
    """
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(num_episodes), desc="Q-Learning"):
        env.reset()
        s = env.state()
        pi = epsilon_greedy_policy(Q, epsilon)
        old_score = env.score()

        while not env.is_game_over():
            a = np.random.choice(num_actions, p=pi[s])
            env.step(a)
            r = env.score() - old_score
            old_score = env.score()
            s_prime = env.state()

            # mise à jour Q-learning
            max_q_next = np.max(Q[s_prime]) if not env.is_game_over() else 0
            target = r + gamma * max_q_next
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_prime

    final_policy = epsilon_greedy_policy(Q, epsilon)
    return final_policy, Q


def expected_sarsa(env, num_episodes=10000, gamma=0.99, alpha=0.1, epsilon=0.1):
    """
    Implémentation de Expected SARSA (TD on-policy)

    Args:
        env: environnement avec .reset(), .state(), .step(a), .score(), .is_game_over(), .num_states(), .num_actions()
        num_episodes: nombre d’épisodes
        gamma: facteur de discount
        alpha: taux d’apprentissage
        epsilon: taux d’exploration

    Returns:
        pi: politique finale ε-greedy
        Q: estimation des valeurs Q(s, a)
    """
    num_states = env.num_states()
    num_actions = env.num_actions()
    Q = np.zeros((num_states, num_actions))

    for _ in tqdm(range(num_episodes), desc="Expected SARSA"):
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

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
                break

            pi_prime = epsilon_greedy_policy(Q, epsilon)
            expected_q = np.dot(pi_prime[s_prime], Q[s_prime])
            target = r + gamma * expected_q
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_prime

    final_policy = epsilon_greedy_policy(Q, epsilon)
    return final_policy, Q
