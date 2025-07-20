import os
import time
import pandas as pd
from tqdm import tqdm
from itertools import product

# Agents Monte Carlo
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control,
    monte_carlo_es,
    off_policy_mc_control
)

# Imports des Environnements
from environments.line_world_env import LineWorldEnv
from environments.grid_world_env import GridWorldEnv
from environments.monty_hall_lv1_env import MontyHallEnv
from environments.monty_hall_lv2_env import MontyHallEnvLv2
from environments.rps_game_env import RPSGameEnv

OUTPUT_DIR = "../Reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGENTS = {
    "on_policy_mc": on_policy_first_visit_mc_control,
    "monte_carlo_es": monte_carlo_es,
    "off_policy_mc": off_policy_mc_control
}

ENVIRONMENTS = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "monty_hall_lv1": MontyHallEnv,
    "monty_hall_lv2": MontyHallEnvLv2,
    "rps_game": RPSGameEnv
}

GAMMAS = [0.3, 0.5, 0.7, 0.9]
EPSILONS = [0.1, 0.3, 0.5, 0.9]
MAX_STEPS = [100]
EPISODES = [500]


def evaluate_policy(env, policy, max_steps=100):
    rewards, steps_list = [], []
    for _ in range(10):
        try:
            env.reset_to(env.get_state())  # Pour Monty Hall
        except:
            env.reset()

        state = env.get_state()
        total_reward, steps = 0, 0

        while not env.is_game_over() and steps < max_steps:
            action = policy.get(state, 0)
            env.step(action)
            total_reward += env.score()
            state = env.get_state()
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)
    return sum(rewards) / len(rewards), rewards, sum(steps_list) / len(steps_list)


def run_monte_carlo_agents():
    results = []

    for agent_name, agent_func in AGENTS.items():
        print(f"\n🔍 Agent : {agent_name}")
        for env_name, EnvCls in tqdm(ENVIRONMENTS.items(), desc="🌍 Environnements"):
            for gamma, epsilon, max_step, episodes in product(GAMMAS, EPSILONS, MAX_STEPS, EPISODES):
                try:
                    env = EnvCls()

                    # Paramètres dynamiques en fonction de l'agent
                    kwargs = {
                        "gamma": gamma,
                        "episodes": episodes
                    }

                    if agent_name == "on_policy_mc":
                        kwargs["epsilon"] = epsilon
                    elif agent_name in ["monte_carlo_es", "off_policy_mc"]:
                        kwargs["max_steps"] = max_step

                    print(
                        f"→ {agent_name} sur {env_name} | γ={gamma}, ε={epsilon}, steps={max_step}, episodes={episodes}")
                    start = time.time()
                    pi, Q, steps_list = agent_func(env, **kwargs)
                    elapsed = round(time.time() - start, 2)

                    mean_score, all_scores, mean_steps = evaluate_policy(env, policy=pi, max_steps=max_step)
                    std_score = pd.Series(all_scores).std()

                    results.append({
                        "agent": agent_name,
                        "env": env_name,
                        "gamma": gamma,
                        "epsilon": epsilon,
                        "episodes": episodes,
                        "max_steps": max_step,
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "mean_steps": mean_steps,
                        "time": elapsed
                    })

                except Exception as e:
                    print(f"❌ Erreur {agent_name} sur {env_name} : {e}")
                    results.append({
                        "agent": agent_name,
                        "env": env_name,
                        "gamma": gamma,
                        "epsilon": epsilon,
                        "episodes": episodes,
                        "max_steps": max_step,
                        "mean_score": None,
                        "std_score": None,
                        "mean_steps": None,
                        "time": None,
                        "error": str(e)
                    })

    # Export Excel
    df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "mc_agents_comparison.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {output_path}")


if __name__ == "__main__":
    t0 = time.time()
    run_monte_carlo_agents()
    print(f"\n⏱️ Script terminé en {round(time.time() - t0, 2)} secondes.")
