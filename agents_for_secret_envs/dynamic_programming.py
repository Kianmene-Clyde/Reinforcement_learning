def policy_iteration(env, gamma=0.99, theta=1e-6):
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_rewards = env.num_rewards()

    V = [0.0 for _ in range(num_states)]
    policy = [0 for _ in range(num_states)]

    is_terminal = [len(env.available_actions()) == 0 for _ in range(num_states)]

    def compute_action_value(s, a, V):
        value = 0.0
        for s_p in range(num_states):
            for r_i in range(num_rewards):
                prob = env.p(s, a, s_p, r_i)
                reward = env.reward(r_i)
                value += prob * (reward + gamma * V[s_p])
        return value

    while True:
        # Évaluation de la politique
        while True:
            delta = 0
            for s in range(num_states):
                if is_terminal[s]:
                    continue
                a = policy[s]
                v = V[s]
                V[s] = compute_action_value(s, a, V)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Amélioration de la politique
        policy_stable = True
        for s in range(num_states):
            if is_terminal[s]:
                continue
            old_action = policy[s]
            best_action = max(
                range(num_actions),
                key=lambda a: compute_action_value(s, a, V)
            )
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return policy


def value_iteration(env, gamma=0.99, theta=1e-6):
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_rewards = env.num_rewards()

    V = [0.0 for _ in range(num_states)]
    policy = [0 for _ in range(num_states)]

    is_terminal = [len(env.available_actions()) == 0 for _ in range(num_states)]

    def compute_action_value(s, a, V):
        value = 0.0
        for s_p in range(num_states):
            for r_i in range(num_rewards):
                prob = env.p(s, a, s_p, r_i)
                reward = env.reward(r_i)
                value += prob * (reward + gamma * V[s_p])
        return value

    while True:
        delta = 0
        for s in range(num_states):
            if is_terminal[s]:
                continue
            v = V[s]
            V[s] = max(compute_action_value(s, a, V) for a in range(num_actions))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    for s in range(num_states):
        if is_terminal[s]:
            continue
        policy[s] = max(
            range(num_actions),
            key=lambda a: compute_action_value(s, a, V)
        )

    return policy
