def policy_iteration(env, gamma=0.99, theta=1e-6):
    V = {s: 0.0 for s in env.get_states()}
    policy = {s: (env.get_actions(s)[0] if env.get_actions(s) else None) for s in env.get_states()}

    while True:
        # Évaluation de la politique actuelle
        while True:
            delta = 0
            for s in env.get_states():
                if env.is_terminal(s):
                    continue
                v = V[s]
                a = policy[s]
                V[s] = sum(prob * (reward + gamma * V[s_prime])
                           for prob, s_prime, reward in env.get_transitions(s, a))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Amélioration de la politique
        policy_stable = True
        for s in env.get_states():
            if env.is_terminal(s):
                continue
            old_action = policy[s]
            policy[s] = max(env.get_actions(s), key=lambda a: sum(
                prob * (reward + gamma * V[s_prime])
                for prob, s_prime, reward in env.get_transitions(s, a)
            ))
            if old_action != policy[s]:
                policy_stable = False

        if policy_stable:
            break

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-5):
    V = {s: 0.0 for s in env.get_states()}
    policy = {}

    while True:
        delta = 0
        for s in env.get_states():
            if env.is_terminal(s):
                continue
            v = V[s]
            V[s] = max(
                sum(p * (r + gamma * V[s_]) for p, s_, r in env.get_transitions(s, a))
                for a in env.get_actions(s)
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    for s in env.get_states():
        if env.is_terminal(s):
            continue
        best_action = max(
            env.get_actions(s),
            key=lambda a: sum(p * (r + gamma * V[s_]) for p, s_, r in env.get_transitions(s, a))
        )
        policy[s] = best_action

    return policy, V
