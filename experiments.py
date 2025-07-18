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

# imports des environnements
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
    "gamma": [0.9, 0.99],
    "alpha": [0.1, 0.5],
    "epsilon": [0.1, 0.2],
    "theta": [1e-4],
    "planning_steps": [5],
    "kappa": [1e-4],
    "episodes": [1000]
}

OUTPUT_DIR = "../Reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []


def evaluate_policy(env, policy):
    rewards = []
    for _ in range(10):
        env.reset()
        state = env.get_state()
        total = 0
        step = 0
        max_steps = 1000

        while not env.is_game_over() and step < max_steps:
            if isinstance(policy, dict):
                action = policy.get(state, 0)
            else:
                action = int(policy[state].argmax())
            env.step(action)
            total += env.score()
            state = env.get_state()
            step += 1

        if step >= max_steps:
            print("\n Épisode bloqué détecté dans evaluate_policy (max_steps atteint)")

        rewards.append(total)
    return sum(rewards) / len(rewards), rewards


def run_experiments():
    xlsx_path = os.path.join(OUTPUT_DIR, "global_comparison.xlsx")

    for env_name, env_cls in tqdm(ENVIRONMENTS.items(), desc="Environnements"):
        env = env_cls()
        env_results = []

        for agent_name, agent_func in tqdm(AGENTS.items(), desc=f"Agents ({env_name})", leave=False):
            param_grid = list(product(
                HYPERPARAM_GRID["gamma"],
                HYPERPARAM_GRID["alpha"],
                HYPERPARAM_GRID["epsilon"],
                HYPERPARAM_GRID["theta"],
                HYPERPARAM_GRID["planning_steps"],
                HYPERPARAM_GRID["kappa"],
                HYPERPARAM_GRID["episodes"]
            ))
            print(f"\n Entrainement de l'agent {agent_name} sur l'environnement {env_name}...")

            for gamma, alpha, epsilon, theta, planning_steps, kappa, episodes in (
                    tqdm(param_grid, desc=f"Hyperparams ({agent_name})", leave=False)):

                print(f"\n Hyperparamètres de {agent_name} : "
                      f"gamma={gamma}, alpha={alpha}, epsilon={epsilon}, "
                      f"theta={theta}, planning_steps={planning_steps}, "
                      f"kappa={kappa}, episodes={episodes}")

                kwargs = dict(
                    gamma=gamma,
                    alpha=alpha,
                    epsilon=epsilon,
                    theta=theta,
                    planning_steps=planning_steps,
                    kappa=kappa,
                    episodes=episodes
                )

                try:
                    start_time = time.time()

                    if agent_name in ["policy_iteration", "value_iteration"]:
                        policy, _ = agent_func(env, gamma=gamma, theta=theta)

                    elif agent_name == "mc_on_policy":
                        policy, _ = agent_func(env, episodes=episodes, gamma=gamma, epsilon=epsilon)

                    elif agent_name == "mc_es":
                        policy, _ = agent_func(env, episodes=episodes, gamma=gamma)

                    elif agent_name == "mc_off_policy":
                        policy, _ = agent_func(env, episodes=episodes, gamma=gamma)

                    elif agent_name in ["sarsa", "q_learning", "expected_sarsa"]:
                        policy, _ = agent_func(env, episodes=episodes, gamma=gamma,
                                               alpha=alpha, epsilon=epsilon)

                    elif agent_name == "dyna_q":
                        policy, _ = agent_func(env, episodes=episodes, gamma=gamma,
                                               alpha=alpha, epsilon=epsilon,
                                               planning_steps=planning_steps)

                    elif agent_name == "dyna_q_plus":
                        policy, _ = agent_func(env, episodes=episodes, gamma=gamma,
                                               alpha=alpha, epsilon=epsilon,
                                               planning_steps=planning_steps, kappa=kappa)

                    else:
                        raise ValueError(f"Agent non supporté: {agent_name}")

                    elapsed_time = time.time() - start_time

                    mean_score, scores = evaluate_policy(env, policy)
                    std_score = pd.Series(scores).std()

                    env_results.append({
                        "agent": agent_name,
                        "env": env_name,
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "time": round(elapsed_time, 2),
                        **kwargs
                    })

                except Exception as e:
                    print(f"\n Erreur avec {agent_name} sur {env_name} : {e}")

        # Export après chaque environnement
        df_env = pd.DataFrame(env_results)
        if os.path.exists(xlsx_path):
            with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df_env.to_excel(writer, sheet_name=f"{env_name}", index=False)
        else:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_env.to_excel(writer, sheet_name=f"{env_name}", index=False)

        print(f"\n Résultats exportés pour {env_name} dans {xlsx_path}")

        all_results.extend(env_results)

    # Export global et best params
    df_all = pd.DataFrame(all_results)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_all.to_excel(writer, sheet_name="RésuméGlobal", index=False)
        best_params = df_all.loc[df_all.groupby(["agent", "env"])["mean_score"].idxmax()]
        best_params.to_excel(writer, sheet_name="BestParams", index=False)

    print("\n Tous les résultats ont été sauvegardés environnement par environnement.")


if __name__ == "__main__":
    start = time.time()
    run_experiments()
    elapsed = time.time() - start
    print(f"\n Expériences terminées en {elapsed:.2f} secondes.")
