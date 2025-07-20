import time
import pandas as pd
import os
import inspect
import itertools

# Agents
from agents_for_secret_envs.monte_carlo_methods import monte_carlo_es, on_policy_first_visit_mc_control, \
    off_policy_mc_control
from agents_for_secret_envs.planning_methods import dyna_q
from agents_for_secret_envs.temporal_difference_methods import q_learning, sarsa, expected_sarsa

# Environnements
from environments.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3

# Agents & Envs
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

# Grille d'hyperparamètres
EPISODES = [100]
GAMMAS = [0.90, 0.95, 0.98, 0.99]
ALPHAS = [0.1, 0.3, 0.5, 0.7]
EPSILONS = [0.05, 0.1, 0.2, 0.3]
PLANNING_STEPS = [0, 5, 10, 20]
KAPPAS = [0.0, 0.0001, 0.001, 0.01]  # Utilisé uniquement pour Dyna-Q+

OUTPUT_DIR = "SecretReports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []
env_results_dict = {}


def filter_hyperparams(agent_func, full_params):
    sig = inspect.signature(agent_func)
    return {k: v for k, v in full_params.items() if k in sig.parameters}


def get_state_from_env(env):
    for attr in ["state_id", "get_state", "state", "observation", "_state", "_get_obs"]:
        if hasattr(env, attr):
            val = getattr(env, attr)
            return val() if callable(val) else val
    return 0


def evaluate_policy(env, policy):
    rewards, steps_list = [], []
    for _ in range(10):
        env.reset()
        try:
            state = get_state_from_env(env)
        except:
            return 0, [], 0

        total, step = 0, 0
        while not env.is_game_over() and step < 1000:
            try:
                available = list(env.available_actions()) if hasattr(env, "available_actions") else None
                action = policy.get(state, 0) if isinstance(policy, dict) else int(policy[state].argmax())
                if available and action not in available:
                    action = available[0]
                env.step(action)
                total += env.score()
                step += 1
                state = get_state_from_env(env)
            except:
                break

        rewards.append(total)
        steps_list.append(step)

    return sum(rewards) / len(rewards), rewards, sum(steps_list) / len(steps_list)


def run_experiments():
    output_path = os.path.join(OUTPUT_DIR, "secret_comparison_full.xlsx")

    for env_name, EnvClass in ENVIRONMENTS.items():
        env = EnvClass()
        env_results = []

        for agent_name, agent_func in AGENTS.items():
            hyper_grid = list(itertools.product(EPISODES, GAMMAS, ALPHAS, EPSILONS, PLANNING_STEPS, KAPPAS))

            for ep, gamma, alpha, epsilon, planning_steps, kappa in hyper_grid:
                HYPERPARAMS = {
                    "episodes": ep,
                    "gamma": gamma,
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "planning_steps": planning_steps,
                    "kappa": kappa,  # sera ignoré si l'agent ne le supporte pas
                }

                try:
                    filtered_params = filter_hyperparams(agent_func, HYPERPARAMS)
                    env.reset()
                    print(f"\n⏳ {agent_name} sur {env_name} | Params: {filtered_params}")
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

                except Exception as e:
                    print(f"❌ Erreur pour {agent_name} sur {env_name} avec {filtered_params} : {e}")

        df_env = pd.DataFrame(env_results)
        env_results_dict[env_name] = df_env

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
                print(f"⚠️ Erreur BestParams : {e}")
        else:
            print("⚠️ Aucun résultat à inclure dans BestParams.")

    print(f"\n✅ Résultats sauvegardés dans : {output_path}")


if __name__ == "__main__":
    start = time.time()
    run_experiments()
    print(f"\n⏱️ Durée totale : {time.time() - start:.2f} secondes.")
