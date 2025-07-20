import time
import pandas as pd
import os
import inspect

# Import des agents pour environnements secrets
from agents_for_secret_envs.monte_carlo_methods import monte_carlo_es, on_policy_first_visit_mc_control, \
    off_policy_mc_control
from agents_for_secret_envs.planning_methods import dyna_q
from agents_for_secret_envs.temporal_difference_methods import q_learning, sarsa, expected_sarsa

# Import des environnements secrets
from environments.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3

# Dictionnaire des agents
AGENTS = {
    "mc_es": monte_carlo_es,
    "mc_on_policy": on_policy_first_visit_mc_control,
    "mc_off_policy": off_policy_mc_control,
    "q_learning": q_learning,
    "sarsa": sarsa,
    "expected_sarsa": expected_sarsa,
    "dyna_q": dyna_q,
}

ENVIRONMENTS = {
    "SecretEnv0": SecretEnv0,
    "SecretEnv1": SecretEnv1,
    "SecretEnv2": SecretEnv2,
    "SecretEnv3": SecretEnv3,
}

# Hyperparamètres fixes
HYPERPARAMS = {
    "episodes": 100,
    "gamma": 0.9,
    "epsilon": 0.1,
    "alpha": 0.5,
    "planning_steps": 10,
}

OUTPUT_DIR = "../SecretReports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []
env_results_dict = {}


def filter_hyperparams(agent_func, full_params):
    sig = inspect.signature(agent_func)
    return {k: v for k, v in full_params.items() if k in sig.parameters}


def get_state_from_env(env):
    if hasattr(env, "get_state"):
        return env.get_state()
    elif hasattr(env, "state") and not callable(env.state):
        return env.state
    elif hasattr(env, "state") and callable(env.state):
        return env.state()
    elif hasattr(env, "observation"):
        return env.observation
    else:
        raise AttributeError(
            f"L'environnement {env.__class__.__name__} ne fournit pas de méthode ou attribut d'état connu.")


def evaluate_policy(env, policy):
    rewards, steps_list = [], []
    for _ in range(10):
        env.reset()
        try:
            state = get_state_from_env(env)
        except Exception as e:
            print(f"Impossible de récupérer l'état initial : {e}")
            return 0, [], 0

        total, step = 0, 0
        while not env.is_game_over() and step < 1000:
            try:
                action = policy.get(state, 0) if isinstance(policy, dict) else int(policy[state].argmax())
            except Exception as e:
                print(f"Erreur lors du choix de l'action : {e}")
                break
            env.step(action)
            print(f"Step {step}, Score actuel: {env.score()}")
            try:
                state = get_state_from_env(env)
            except:
                break
            total += env.score()
            step += 1
        rewards.append(total)
        steps_list.append(step)
    return sum(rewards) / len(rewards), rewards, sum(steps_list) / len(steps_list)


def run_experiments():
    output_path = os.path.join(OUTPUT_DIR, "secret_comparison_full.xlsx")

    for env_name, EnvClass in ENVIRONMENTS.items():
        env = EnvClass()
        env_results = []

        for agent_name, agent_func in AGENTS.items():
            try:
                filtered_params = filter_hyperparams(agent_func, HYPERPARAMS)
                env.reset()
                print(f"\nLancement : {agent_name} sur {env_name}")
                start_time = time.time()
                result = agent_func(env, **filtered_params)
                elapsed_time = round(time.time() - start_time, 2)

                policy = result.get("policy") if isinstance(result, dict) else result[0]
                mean_score, all_scores, mean_steps = evaluate_policy(env, policy)
                std_score = pd.Series(all_scores).std()

                res = {
                    "agent": agent_name,
                    "env": env_name,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "mean_steps": mean_steps,
                    "time": elapsed_time,
                    **filtered_params,
                }

                env_results.append(res)
                all_results.append(res)
                print(f"Succès {agent_name} sur {env_name}")

            except Exception as e:
                print(f"Erreur pour {agent_name} sur {env_name} : {e}")

        df_env = pd.DataFrame(env_results)
        env_results_dict[env_name] = df_env

    # Écriture dans un seul fichier Excel
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        for env_name, df in env_results_dict.items():
            df.to_excel(writer, sheet_name=env_name, index=False)

        df_all = pd.DataFrame(all_results)
        df_all.to_excel(writer, sheet_name="RésuméGlobal", index=False)

        if not df_all.empty:
            try:
                best_params = df_all.loc[df_all.groupby(["agent", "env"])["mean_score"].idxmax()]
                best_params.to_excel(writer, sheet_name="BestParams", index=False)
            except Exception as e:
                print(f"Erreur lors de la génération des BestParams : {e}")
        else:
            print("Aucun résultat à inclure dans BestParams.")

    print(f"\nFichier Excel unique généré : {output_path}")


if __name__ == "__main__":
    start = time.time()
    run_experiments()
    print(f"\nTerminé en {time.time() - start:.2f} secondes.")
