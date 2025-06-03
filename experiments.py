import json
import os
from datetime import datetime
from agents.dynamic_programming import policy_iteration
from environments.line_world_env import LineWorldEnv

# === Liste des configurations ===
AGENTS = {
    "policy_iteration": policy_iteration,
    # Ajoute ici les autres agents (ex: "value_iteration": value_iteration)
}

ENVIRONMENTS = {
    "line_world": LineWorldEnv,
    # Ajoute ici d'autres environnements (ex: "grid_world": GridWorldEnv)
}

HYPERPARAMS_GRID = {
    "gamma": [0.9, 0.95, 0.99],
    "theta": [0.01, 0.001, 0.0001]
}

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_policy(env, policy):
    rewards = []
    for _ in range(10):  # 10 épisodes d'évaluation
        state = env.reset()
        total = 0
        while not env.is_terminal(state):
            action = policy.get(state, 0)
            next_state, reward = env.transition(state, action)
            state = next_state
            total += reward
        rewards.append(total)
    return sum(rewards) / len(rewards), rewards


def run_experiments():
    for env_name, env_cls in ENVIRONMENTS.items():
        env = env_cls()

        # Adapter si besoin : ajout dynamique des méthodes
        env.get_states = lambda: env.states
        env.get_actions = lambda s: [0, 1] if s not in env.terminal_states else []
        env.is_terminal = lambda s: s in env.terminal_states
        env.get_transitions = lambda s, a: [(1.0, *env.transition(s, a))]

        for agent_name, agent_func in AGENTS.items():
            for gamma in HYPERPARAMS_GRID["gamma"]:
                for theta in HYPERPARAMS_GRID["theta"]:
                    policy, V = agent_func(env, gamma=gamma, theta=theta)
                    mean_score, scores = evaluate_policy(env, policy)

                    report = {
                        "agent": agent_name,
                        "environment": env_name,
                        "gamma": gamma,
                        "theta": theta,
                        "mean_score": mean_score,
                        "scores": scores,
                        "timestamp": datetime.now().isoformat()
                    }

                    filename = f"{agent_name}_{env_name}_gamma_{gamma}_theta_{theta}.json"
                    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
                        json.dump(report, f, indent=2)

                    print(f"✅ Rapport sauvegardé : {filename}")


if __name__ == "__main__":
    run_experiments()
