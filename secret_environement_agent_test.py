import os
import time
import pandas as pd
from datetime import datetime

from agents_for_secret_envs.monte_carlo_methods import on_policy_first_visit_mc_control, monte_carlo_es, \
    off_policy_mc_control
from agents_for_secret_envs.planning_methods import dyna_q, dyna_q_plus
from agents_for_secret_envs.temporal_difference_methods import q_learning, sarsa, expected_sarsa
from environments.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3

start_time = time.time()

# Liste des environnements
environments = [
    ("SecretEnv0", SecretEnv0),
    ("SecretEnv1", SecretEnv1),
    ("SecretEnv2", SecretEnv2),
    ("SecretEnv3", SecretEnv3),
]

# Liste des agents avec leurs hyperparamètres
agents = [
    ("Q-Learning", lambda env: q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=5000)),
    ("SARSA", lambda env: sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=5000)),
    ("Expected SARSA", lambda env: expected_sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=5000)),
    ("On-policy MC Control", lambda env: on_policy_first_visit_mc_control(env, gamma=0.99, epsilon=0.1, episodes=5000)),
    ("Monte Carlo ES", lambda env: monte_carlo_es(env, gamma=0.99, episodes=5000)),
    ("Off-policy MC", lambda env: off_policy_mc_control(env, gamma=0.99, episodes=5000)),
    ("Dyna-Q", lambda env: dyna_q(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=5000, planning_steps=10)),
    ("Dyna-Q+",
     lambda env: dyna_q_plus(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=5000, planning_steps=10, kappa=0.01)),
]


# Fonction d’évaluation d’une politique
def evaluate_policy(env, policy, max_steps=1000):
    env.reset()
    steps = 0
    while not env.is_game_over() and steps < max_steps:
        state = env.state_id()
        actions = env.available_actions()
        if isinstance(policy, dict):
            action = policy.get(state, actions[0])
        else:
            action = policy[state] if state < len(policy) and policy[state] in actions else actions[0]
        env.step(action)
        steps += 1
    return env.score(), steps


# Lancement des tests
results = []
for env_name, EnvClass in environments:
    env = EnvClass()
    for agent_name, agent_fn in agents:
        print(f"Test de {agent_name} sur {env_name}")
        try:
            policy = agent_fn(env)
            score, steps = evaluate_policy(env, policy)
            results.append({
                "Environnement": env_name,
                "Agent": agent_name,
                "Score final": score,
                "Étapes": steps,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Erreur": "",
            })
        except Exception as e:
            results.append({
                "Environnement": env_name,
                "Agent": agent_name,
                "Score final": "Erreur",
                "Étapes": "-",
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Erreur": str(e),
            })
            print(f"Erreur avec {agent_name} sur {env_name} : {e}")

df = pd.DataFrame(results)
os.makedirs("exports", exist_ok=True)
df.to_excel("exports/secret_envs_results.xlsx", index=False)
print("\n Résultats sauvegardés dans 'exports/secret_envs_results.xlsx'")

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"\n Temps d'exécution total : {minutes} min {seconds} sec")
