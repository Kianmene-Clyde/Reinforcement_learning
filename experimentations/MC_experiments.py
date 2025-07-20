# Rechargement des modules et exécution après le reset de l'environnement

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from agents.monte_carlo_methods import (
    monte_carlo_es,
    on_policy_first_visit_mc_control,
    off_policy_mc_control,
)

from environments.line_world_env import LineWorldEnv
from environments.grid_world_env import GridWorldEnv
from environments.monty_hall_lv1_env import MontyHallEnv
from environments.monty_hall_lv2_env import MontyHallEnvLv2
from environments.rps_game_env import RPSGameEnv

# === Dossiers de sortie
RESULTS_DIR = "MC_Results"
QTABLE_DIR = os.path.join(RESULTS_DIR, "q_tables")
HEATMAP_DIR = os.path.join(RESULTS_DIR, "heatmaps")
os.makedirs(QTABLE_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

# === Agents et Environnements
AGENTS = {
    "mc_es": monte_carlo_es,
    "mc_on_policy": on_policy_first_visit_mc_control,
    "mc_off_policy": off_policy_mc_control,
}

ENVIRONMENTS = {
    "LineWorld": LineWorldEnv(),
    "GridWorld": GridWorldEnv(),
    "MontyHallLv1": MontyHallEnv(),
    "MontyHallLv2": MontyHallEnvLv2(),
    "RPS": RPSGameEnv(),
}

# === Hyperparamètres par agent
AGENT_PARAMS = {
    "mc_es": {"episodes": 1000, "gamma": 0.95, "max_steps": 100},
    "mc_on_policy": {"episodes": 1000, "gamma": 0.95, "epsilon": 0.1},
    "mc_off_policy": {"episodes": 1000, "gamma": 0.95, "max_steps": 100},
}


def save_q_and_heatmap(q_table, env_name, agent_name):
    df_q = pd.DataFrame(q_table)

    # Sauvegarde CSV
    q_path = os.path.join(QTABLE_DIR, f"Q_{env_name}_{agent_name}.csv")
    df_q.to_csv(q_path, index_label="État")

    # Création de la heatmap inversée (Actions en ligne, États en colonne)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df_q.T, annot=True, cmap="coolwarm_r", cbar=True, vmin=np.min(q_table), vmax=np.max(q_table), fmt=".2f")

    # Titre et axes
    plt.title(f"Q-table Heatmap | {env_name} | {agent_name}")
    plt.xlabel("État")
    plt.ylabel("Action")

    if env_name == "LineWorld" and df_q.shape[1] == 2:
        ax.set_yticklabels(["Gauche", "Droite"])

    plt.tight_layout()

    # Sauvegarde
    heatmap_path = os.path.join(HEATMAP_DIR, f"heatmap_{env_name}_{agent_name}.png")
    plt.savefig(heatmap_path)
    plt.close()

    return q_path, heatmap_path


def run_all_experiments():
    results = []
    for env_name, env in ENVIRONMENTS.items():
        for agent_name, agent_func in AGENTS.items():
            print(f"🔄 Entraînement de {agent_name} sur {env_name}")
            try:
                params = AGENT_PARAMS[agent_name]
                result = agent_func(env, **params)
                policy, q_table, steps = result

                q_path, hm_path = save_q_and_heatmap(q_table, env_name, agent_name)

                results.append({
                    "env": env_name,
                    "agent": agent_name,
                    "episodes": params.get("episodes"),
                    "gamma": params.get("gamma"),
                    "epsilon": params.get("epsilon", None),
                    "max_steps": params.get("max_steps", None),
                    "mean_steps": np.mean(steps),
                    "q_table_csv": q_path,
                    "heatmap_png": hm_path,
                })
            except Exception as e:
                print(f"❌ Erreur pour {agent_name} sur {env_name} : {e}")

    df = pd.DataFrame(results)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    df.to_csv(summary_path, index=False)
    return df

# Exécution
df_results = run_all_experiments()

df_results