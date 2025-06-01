def policy_iteration(env, gamma=0.99, theta=1e-4):
    states = env.get_states()
    V = {s: 0 for s in states}
    policy = {s: [env.get_actions(s)[0]] for s in states if env.get_actions(s)}

    while True:
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for s in states:
                if s in env.terminal_states:
                    continue
                v = V[s]
                a = policy[s][0]
                next_s, r = env.transition(s, a)
                V[s] = r + gamma * V[next_s]
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        policy_stable = True
        for s in states:
            if s in env.terminal_states:
                continue
            old_action = policy[s][0]
            action_values = {}
            for a in env.get_actions(s):
                next_s, r = env.transition(s, a)
                action_values[a] = r + gamma * V[next_s]
            best_action = max(action_values, key=action_values.get)
            policy[s] = [best_action]
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, V
