import os
import time
import pandas as pd
from tqdm import tqdm
from itertools import product

# Import des agents TD
from agents.temporal_difference_methods import sarsa, q_learning, expected_sarsa

# Import des environnements
from environments.line_world_env import LineWorldEnv
from environments.grid_world_env import GridWorldEnv
from environments.monty_hall_lv1_env import MontyHallEnv
from environments.monty_hall_lv2_env import MontyHallEnvLv2
from environments.rps_game_env import RPSGameEnv

# Configuration des agents TD
AGENTS = {
    "sarsa": sarsa,
    "q_learning": q_learning,
    "expected_sarsa": expected_sarsa
}

ENVIRONMENTS = {
    "line_world": LineWorldEnv,
    "monty_hall_lv1": MontyHallEnv,
    "monty_hall_lv2": MontyHallEnvLv2,
    "rps_game": RPSGameEnv,
    "grid_world": GridWorldEnv
}

# Hyperparamètres
GAMMAS = [0.3, 0.5, 0.7, 0.9]
ALPHAS = [0.01, 0.1, 0.3, 0.5]
EPSILONS = [0.1, 0.3, 0.5, 0.9]

EPISODES = 500
OUTPUT_DIR = "../Reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Évaluation de la politique obtenue
def evaluate_policy(env, policy):
    rewards, steps_list = [], []
    for _ in range(10):
        env.reset()
        state = env.get_state()
        total, step = 0, 0
        while not env.is_game_over() and step < 100:
            action = policy[state].argmax() if hasattr(policy[state], "argmax") else int(policy.get(state, 0))
            if action is None:
                break
            env.step(action)
            total += env.score()
            state = env.get_state()
            step += 1
        rewards.append(total)
        steps_list.append(step)
    return sum(rewards) / len(rewards), rewards, sum(steps_list) / len(steps_list)


# Entraînement
def run_all_td_agents():
    results = []
    for agent_name, agent_func in AGENTS.items():
        print(f"\nTest de l'agent : {agent_name}")
        for env_name, EnvCls in tqdm(ENVIRONMENTS.items(), desc="Environnements"):
            for gamma, alpha, epsilon in product(GAMMAS, ALPHAS, EPSILONS):
                try:
                    env = EnvCls()
                    print(f"{agent_name} sur {env_name} | γ={gamma}, α={alpha}, ε={epsilon}")

                    kwargs = {
                        "gamma": gamma,
                        "alpha": alpha,
                        "epsilon": epsilon,
                        "episodes": EPISODES
                    }

                    start = time.time()
                    policy, _, steps = agent_func(env, **kwargs)
                    elapsed = round(time.time() - start, 2)

                    mean_score, all_scores, mean_steps = evaluate_policy(env, policy)
                    std_score = pd.Series(all_scores).std()

                    results.append({
                        "agent": agent_name,
                        "env": env_name,
                        "gamma": gamma,
                        "alpha": alpha,
                        "epsilon": epsilon,
                        "episodes": EPISODES,
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "mean_steps": mean_steps,
                        "time": elapsed
                    })

                except Exception as e:
                    print(f"Erreur {agent_name} sur {env_name} : {e}")
                    results.append({
                        "agent": agent_name,
                        "env": env_name,
                        "gamma": gamma,
                        "alpha": alpha,
                        "epsilon": epsilon,
                        "episodes": EPISODES,
                        "mean_score": None,
                        "std_score": None,
                        "mean_steps": None,
                        "time": None,
                        "error": str(e)
                    })

    # Sauvegarde des résultats
    df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "td_agents_comparison.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\nRésultats sauvegardés dans : {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    run_all_td_agents()
    elapsed_time = round(time.time() - start_time, 2)
    print(f"\nExécution terminée en {elapsed_time} secondes.")
