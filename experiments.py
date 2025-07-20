import os
import time
import pandas as pd
from datetime import datetime
from itertools import product
from tqdm import tqdm

# Imports des agents
from agents.dynamic_programming import policy_iteration, value_iteration
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control, monte_carlo_es, off_policy_mc_control
)
from agents.temporal_difference_methods import sarsa, q_learning, expected_sarsa
from agents.planning_methods import dyna_q, dyna_q_plus

# Imports des environnements
from environments.grid_world_env_headless import GridWorldEnvHeadless
from environments.line_world_env import LineWorldEnv
from environments.monty_hall_lv1_env import MontyHallEnv
from environments.monty_hall_lv2_env import MontyHallEnvLv2
from environments.rps_game_env import RPSGameEnv

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
    "monty_hall_lv1": MontyHallEnv,
    "monty_hall_lv2": MontyHallEnvLv2,
    "rps_game": RPSGameEnv,
    "grid_world": GridWorldEnvHeadless
}

HYPERPARAM_GRID = {
    "gamma": [0.90, 0.95, 0.98, 0.99],
    "alpha": [0.01, 0.05, 0.1, 0.5],
    "epsilon": [0.01, 0.05, 0.1, 0.2],
    "theta": [0.001, 0.0001, 0.00001, 0.000001],
    "planning_steps": [5, 10, 20, 50],
    "kappa": [0.0, 0.0001, 0.001, 0.1],
    "episodes": [500]
}

OUTPUT_DIR = "../Reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []


def evaluate_policy(env, policy):
    rewards = []
    steps_list = []
    for _ in range(10):
        env.reset()
        state = env.get_state()
        total, step = 0, 0
        while not env.is_game_over() and step < 1000:
            action = policy.get(state, 0) if isinstance(policy, dict) else int(policy[state].argmax())
            env.step(action)
            total += env.score()
            state = env.get_state()
            step += 1
        rewards.append(total)
        steps_list.append(step)
    return sum(rewards) / len(rewards), rewards, sum(steps_list) / len(steps_list)


def get_param_combinations(agent_name):
    keys = []
    if agent_name in ["policy_iteration", "value_iteration"]:
        keys = ["gamma", "theta"]
    elif agent_name in ["mc_on_policy"]:
        keys = ["gamma", "epsilon", "episodes"]
    elif agent_name in ["mc_es", "mc_off_policy"]:
        keys = ["gamma", "episodes"]
    elif agent_name in ["sarsa", "q_learning", "expected_sarsa"]:
        keys = ["gamma", "alpha", "epsilon", "episodes"]
    elif agent_name == "dyna_q":
        keys = ["gamma", "alpha", "epsilon", "planning_steps", "episodes"]
    elif agent_name == "dyna_q_plus":
        keys = ["gamma", "alpha", "epsilon", "planning_steps", "kappa", "episodes"]
    return keys, list(product(*(HYPERPARAM_GRID[k] for k in keys)))


def run_experiments():
    xlsx_path = os.path.join(OUTPUT_DIR, "global_comparison.xlsx")
    for env_name, env_cls in tqdm(ENVIRONMENTS.items(), desc="Environnements"):
        env = env_cls()
        env_results = []

        for agent_name, agent_func in tqdm(AGENTS.items(), desc=f"Agents ({env_name})", leave=False):
            keys, param_grid = get_param_combinations(agent_name)
            for param_tuple in tqdm(param_grid, desc=f"{agent_name}", leave=False):
                params = dict(zip(keys, param_tuple))
                try:
                    start_time = time.time()
                    if agent_name in ["policy_iteration", "value_iteration"]:
                        policy, _ = agent_func(env, **params)
                    else:
                        policy, _, steps = agent_func(env, **params)
                    elapsed = time.time() - start_time
                    mean_score, scores, mean_steps = evaluate_policy(env, policy)
                    std_score = pd.Series(scores).std()
                    result = {
                        "agent": agent_name,
                        "env": env_name,
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "mean_steps": mean_steps,
                        "time": round(elapsed, 2),
                        **params
                    }
                    env_results.append(result)
                except Exception as e:
                    print(f"\nErreur avec {agent_name} sur {env_name} : {e}")

        df_env = pd.DataFrame(env_results)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a" if os.path.exists(xlsx_path) else "w",
                            if_sheet_exists="replace") as writer:
            df_env.to_excel(writer, sheet_name=f"{env_name}", index=False)
        all_results.extend(env_results)

    df_all = pd.DataFrame(all_results)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_all.to_excel(writer, sheet_name="RésuméGlobal", index=False)
        best_params = df_all.loc[df_all.groupby(["agent", "env"])["mean_score"].idxmax()]
        best_params.to_excel(writer, sheet_name="BestParams", index=False)
    print("\nTous les résultats ont été sauvegardés.")


if __name__ == "__main__":
    start = time.time()
    run_experiments()
    print(f"\nExpériences terminées en {time.time() - start:.2f} secondes.")
