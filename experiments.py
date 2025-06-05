import os
import time
import pandas as pd
from datetime import datetime
from itertools import product
from tqdm import tqdm

# === Imports des agents ===
from agents.dynamic_programming import policy_iteration, value_iteration
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control, monte_carlo_es, off_policy_mc_control
)
from agents.temporal_difference_methods import sarsa, q_learning, expected_sarsa
from agents.planning_methods import dyna_q, dyna_q_plus

# === Imports des environnements ===
from environments.line_world_env import LineWorldEnv
from environments.grid_world_env import GridWorldEnv
from environments.monty_hall_lv1_env import MontyHallEnv
from environments.monty_hall_lv2_env import MontyHallEnvLv2
from environments.rps_game_env import RPSGameEnv

# === Fonctions utilitaires ===
from Utils.save_load_policy import save_policy

# === Liste des configurations ===
AGENTS = {
    "policy_iteration": policy_iteration,
    "value_iteration": value_iteration,
    "mc_on_policy": on_policy_first_visit_mc_control,
    "mc_es": monte_carlo_es,
    "mc_off_policy": off_policy_mc_control,
    "sarsa": sarsa,
    "q_learning": q_learning,
    "expected_sarsa": expected_sarsa,
    "dyna_q": dyna_q,
    "dyna_q_plus": dyna_q_plus,
}

ENVIRONMENTS = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "monty_hall_lv1": MontyHallEnv,
    "monty_hall_lv2": MontyHallEnvLv2,
    "rps_game": RPSGameEnv
}

HYPERPARAM_GRID = {
    "gamma": [0.9, 0.99],
    "alpha": [0.1, 0.5],
    "epsilon": [0.1, 0.2],
    "theta": [1e-4],
    "planning_steps": [5],
    "kappa": [1e-4],
    "num_episodes": [1000]
}

OUTPUT_DIR = "Reports"
POLICY_DIR = os.path.join(OUTPUT_DIR, "Policies")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(POLICY_DIR, exist_ok=True)
all_results = []


def evaluate_policy(env, policy):
    rewards = []
    for _ in range(10):
        state = env.reset()
        total = 0
        while not env.is_terminal(state):
            action = policy.get(state, 0) if isinstance(policy, dict) else policy[state].argmax()
            next_state, reward = env.transition(state, action)
            state = next_state
            total += reward
        rewards.append(total)
    return sum(rewards) / len(rewards), rewards


def run_experiments():
    for env_name, env_cls in tqdm(ENVIRONMENTS.items(), desc="Environnements"):
        env = env_cls()

        for agent_name, agent_func in tqdm(AGENTS.items(), desc=f"Agents ({env_name})", leave=False):
            param_grid = list(product(
                HYPERPARAM_GRID["gamma"],
                HYPERPARAM_GRID["alpha"],
                HYPERPARAM_GRID["epsilon"],
                HYPERPARAM_GRID["theta"],
                HYPERPARAM_GRID["planning_steps"],
                HYPERPARAM_GRID["kappa"],
                HYPERPARAM_GRID["num_episodes"]
            ))

            for gamma, alpha, epsilon, theta, planning_steps, kappa, num_episodes in (
                    tqdm(param_grid, desc=f"Hyperparams ({agent_name})", leave=False)):
                kwargs = {
                    "gamma": gamma,
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "theta": theta,
                    "planning_steps": planning_steps,
                    "kappa": kappa,
                    "num_episodes": num_episodes
                }

                try:
                    if agent_name in ["policy_iteration", "value_iteration"]:
                        policy, _ = agent_func(env, gamma=gamma, theta=theta)
                    elif agent_name == "mc_on_policy":
                        policy, _ = agent_func(env, num_episodes=num_episodes, gamma=gamma, epsilon=epsilon)
                    elif agent_name == "mc_es":
                        policy, _ = agent_func(env, num_episodes=num_episodes, gamma=gamma)
                    elif agent_name == "mc_off_policy":
                        policy, _ = agent_func(env, num_episodes=num_episodes, gamma=gamma)
                    elif agent_name in ["sarsa", "q_learning", "expected_sarsa"]:
                        policy, _ = agent_func(env, num_episodes=num_episodes, gamma=gamma, alpha=alpha,
                                               epsilon=epsilon)
                    elif agent_name == "dyna_q":
                        policy, _ = agent_func(env, num_episodes=num_episodes, gamma=gamma, alpha=alpha,
                                               epsilon=epsilon, planning_steps=planning_steps)
                    elif agent_name == "dyna_q_plus":
                        policy, _ = agent_func(env, num_episodes=num_episodes, gamma=gamma, alpha=alpha,
                                               epsilon=epsilon, planning_steps=planning_steps, kappa=kappa)
                    else:
                        raise ValueError("Agent non pris en charge.")

                    mean_score, scores = evaluate_policy(env, policy)

                    filename = f"policy_{env_name}_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    save_policy(policy, os.path.join(POLICY_DIR, filename))

                    all_results.append({
                        "agent": agent_name,
                        "env": env_name,
                        "mean_score": mean_score,
                        **kwargs
                    })

                except Exception as e:
                    print(f"❌ Erreur avec {agent_name} sur {env_name} : {e}")

    # Résumé global : export .xlsx + graphes
    df = pd.DataFrame(all_results)
    xlsx_path = os.path.join(OUTPUT_DIR, "global_comparison.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Résultats", index=False)

        best_params = df.loc[df.groupby(["agent", "env"])["mean_score"].idxmax()]
        best_params.to_excel(writer, sheet_name="BestParams", index=False)

    print("\n📊 Résumé global exporté en .xlsx avec toutes les politiques sauvegardées.")


if __name__ == "__main__":
    start = time.time()
    run_experiments()
    elapsed = time.time() - start
    print(f"\n✅ Expériences terminées en {elapsed:.2f} secondes.")
