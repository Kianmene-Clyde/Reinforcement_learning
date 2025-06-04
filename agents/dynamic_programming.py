def policy_iteration(env, gamma=0.99, theta=1e-4, max_iterations=1000, verbose=False):
    """
    Policy Iteration for (potentially stochastic) environments.
    env must implement:
        - env.get_states(): list of all states
        - env.get_actions(state): list of possible actions in a state
        - env.get_transitions(state, action): list of (probability, next_state, reward)
        - env.is_terminal(state): True if state is terminal
    """
    states = env.get_states()
    V = {s: 0.0 for s in states}
    policy = {}

    # Init policy randomly
    for s in states:
        if not env.is_terminal(s):
            actions = env.get_actions(s)
            if actions:
                policy[s] = actions[0]

    for iteration in range(max_iterations):
        if verbose:
            print(f"=== Iteration {iteration} ===")

        # --- Policy Evaluation ---
        while True:
            delta = 0.0
            for s in states:
                if env.is_terminal(s):
                    continue
                v = V[s]
                a = policy[s]
                V[s] = sum([
                    p * (r + gamma * V[s_])
                    for (p, s_, r) in env.get_transitions(s, a)
                ])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        policy_stable = True
        for s in states:
            if env.is_terminal(s):
                continue
            old_action = policy[s]
            actions = env.get_actions(s)
            action_values = {}
            for a in actions:
                action_values[a] = sum([
                    p * (r + gamma * V[s_])
                    for (p, s_, r) in env.get_transitions(s, a)
                ])
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            if verbose:
                print("Politique convergente trouvée.")
            break

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-6):
    V = {s: 0.0 for s in env.get_states()}
    policy = {}

    while True:
        delta = 0
        for s in env.get_states():
            if env.is_terminal(s):
                continue
            v = V[s]
            V[s] = max(
                sum(prob * (reward + gamma * V[s_prime])
                    for prob, s_prime, reward in env.get_transitions(s, a))
                for a in env.get_actions(s)
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    for s in env.get_states():
        if env.is_terminal(s):
            continue
        policy[s] = max(
            env.get_actions(s),
            key=lambda a: sum(
                prob * (reward + gamma * V[s_prime])
                for prob, s_prime, reward in env.get_transitions(s, a)
            )
        )

    return V, policy
