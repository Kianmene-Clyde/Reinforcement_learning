from environments.grid_world_runner import GridWorldRunner
from environments.line_world_runner import LineWorldRunner

if __name__ == "__main__":
    print("Choisissez un agent :")
    print("1 - Policy Iteration")
    print("2 - Value Iteration")
    print("3 - Dyna Q")
    print("4 - Dyna Q+")
    print("5 - Sarsa")
    print("6 - Expected Sarsa")
    print("7 - First visit Monte Carlo")
    print("8 - Monte Carlo ES")
    print("9 - Off-policy Monte Carlo")
    print("10 - Q Learning")

    agent_choice = input("Entrée : ").strip()

    print("Choisissez un environnement :")
    print("1 - LineWorld")
    print("2 - GridWorld")
    print("3 - Monty Hall lvl 1")
    print("4 - Monty Hall lvl 2")
    print("5 - RPS")

    env_choice = input("Entrée : ").strip()

    if env_choice == "1":
        if agent_choice == "1":  # Policy Iteration
            runner = LineWorldRunner(agent_name="Policy Iteration")
            runner.run()
        else:
            print("Agent non encore supporté pour LineWorld.")

    elif env_choice == "2":
        if agent_choice == "1":  # Policy Iteration
            runner = GridWorldRunner(agent_name="Policy Iteration")
            runner.run()
        else:
            print("Agent non encore supporté pour GridWorld.")

    else:
        print("Environnement non reconnu.")
