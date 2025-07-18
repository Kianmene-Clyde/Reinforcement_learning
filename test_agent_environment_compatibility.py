import sys
import traceback

from agents.dynamic_programming import policy_iteration, value_iteration
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control,
    monte_carlo_es,
    off_policy_mc_control,
)
from agents.temporal_difference_methods import sarsa, q_learning, expected_sarsa
from agents.planning_methods import dyna_q, dyna_q_plus

from environments.line_world_env import LineWorldEnv
from environments.grid_world_env import GridWorldEnv
from environments.monty_hall_lv1_env import MontyHallEnv
from environments.monty_hall_lv2_env import MontyHallEnvLv2
from environments.rps_game_env import RPSGameEnv


# === Configuration des tests ===
def get_agent_function(agent_name, env_name):
    """Renvoie la fonction agent configurée dynamiquement"""
    if agent_name == "Policy Iteration":
        return lambda env: policy_iteration(env)
    elif agent_name == "Value Iteration":
        return lambda env: value_iteration(env)
    elif agent_name == "First visit Monte Carlo":
        return lambda env: on_policy_first_visit_mc_control(env, episodes=10)
    elif agent_name == "Monte Carlo ES":
        # ⚠️ Réduction des épisodes sur GridWorld
        if env_name == "GridWorld":
            return lambda env: monte_carlo_es(env, episodes=2)
        else:
            return lambda env: monte_carlo_es(env, episodes=10)
    elif agent_name == "Off-policy Monte Carlo":
        return lambda env: off_policy_mc_control(env, episodes=10)
    elif agent_name == "Sarsa":
        return lambda env: sarsa(env, episodes=10)
    elif agent_name == "Q Learning":
        return lambda env: q_learning(env, episodes=10)
    elif agent_name == "Expected Sarsa":
        return lambda env: expected_sarsa(env, episodes=10)
    elif agent_name == "Dyna Q":
        return lambda env: dyna_q(env, episodes=10)
    elif agent_name == "Dyna Q+":
        return lambda env: dyna_q_plus(env, episodes=10)
    else:
        raise ValueError(f"Agent inconnu : {agent_name}")


AGENT_NAMES = [
    "Policy Iteration",
    "Value Iteration",
    "First visit Monte Carlo",
    "Monte Carlo ES",
    "Off-policy Monte Carlo",
    "Sarsa",
    "Q Learning",
    "Expected Sarsa",
    "Dyna Q",
    "Dyna Q+",
]

ENVS = {
    "LineWorld": LineWorldEnv,
    "GridWorld": GridWorldEnv,
    "MontyHall LV1": MontyHallEnv,
    "MontyHall LV2": MontyHallEnvLv2,
    "RPS Game": RPSGameEnv,
}


def test_compatibility():
    print(" Test de compatibilité agent <-> environnement...\n")
    for env_name, env_class in ENVS.items():
        for agent_name in AGENT_NAMES:
            try:
                print(f"Test: Agent '{agent_name}' sur '{env_name}'... ", end="")
                env = env_class()
                agent_func = get_agent_function(agent_name, env_name)
                policy, q = agent_func(env)
                print("Succès")
            except Exception as e:
                print("Échec")
                traceback.print_exc(limit=1, file=sys.stdout)


if __name__ == "__main__":
    test_compatibility()
