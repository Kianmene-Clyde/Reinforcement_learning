import random


def policy_iteration(env, gamma=0.99, theta=1e-6, verbose=False):
    states = env.get_states()
    V = {s: 0.0 for s in states}
    policy = {}

    # Initialiser la politique aléatoirement (valide pour tous les états non terminaux)
    for s in states:
        actions = env.get_actions(s)
        policy[s] = random.choice(actions) if actions else None

    # Pré-calculer les transitions
    transitions = {
        (s, a): env.get_transitions(s, a)
        for s in states for a in env.get_actions(s)
    }

    iterations = 0
    while True:
        # Évaluation de la politique
        while True:
            delta = 0
            for s in states:
                if env.is_terminal(s):
                    continue
                a = policy[s]
                v = V[s]
                V[s] = sum(prob * (reward + gamma * V[s_]) for prob, s_, reward in transitions[(s, a)])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Amélioration de la politique
        policy_stable = True
        for s in states:
            if env.is_terminal(s):
                continue
            old_action = policy[s]
            action_values = {
                a: sum(prob * (reward + gamma * V[s_]) for prob, s_, reward in transitions[(s, a)])
                for a in env.get_actions(s)
            }
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        iterations += 1
        if verbose:
            print(f"[Policy Iteration] Iteration {iterations} terminée.")

        if policy_stable:
            break

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-6, verbose=False):
    states = env.get_states()
    V = {s: 0.0 for s in states}
    transitions = {
        (s, a): env.get_transitions(s, a)
        for s in states for a in env.get_actions(s)
    }

    iterations = 0
    while True:
        delta = 0
        for s in states:
            if env.is_terminal(s):
                continue
            v = V[s]
            V[s] = max(
                sum(prob * (reward + gamma * V[s_]) for prob, s_, reward in transitions[(s, a)])
                for a in env.get_actions(s)
            )
            delta = max(delta, abs(v - V[s]))
        iterations += 1
        if verbose:
            print(f"[Value Iteration] Iteration {iterations} terminée - delta={delta:.8f}")
        if delta < theta:
            break

    policy = {}
    for s in states:
        if env.is_terminal(s):
            continue
        action_values = {
            a: sum(prob * (reward + gamma * V[s_]) for prob, s_, reward in transitions[(s, a)])
            for a in env.get_actions(s)
        }
        policy[s] = max(action_values, key=action_values.get)

    return policy, V
