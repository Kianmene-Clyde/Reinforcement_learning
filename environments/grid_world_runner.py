import pygame
import sys
from Utils.export_results_to_xlsx import export_results
from Utils.save_load_policy import save_policy, load_policy
from agents.dynamic_programming import policy_iteration, value_iteration
from agents.planning_methods import dyna_q, dyna_q_plus
from agents.temporal_difference_methods import sarsa, expected_sarsa, q_learning
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control,
    monte_carlo_es,
    off_policy_mc_control
)

from environments.grid_world_env import GridWorldEnv

CELL_SIZE = 80
WHITE, BLACK, GREY, BLUE, RED, GREEN = (
    (255, 255, 255), (0, 0, 0), (220, 220, 220), (0, 0, 255), (200, 50, 50), (50, 200, 50)
)


class GridWorldRunner:
    def __init__(self, agent_name="Policy Iteration"):
        self.env = GridWorldEnv()
        self.agent_name = agent_name
        self.policy = {}
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 20)
        width = self.env.width * CELL_SIZE
        height = self.env.height * CELL_SIZE + 50
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Grid World - {self.agent_name}")

        self.hyperparams_map = {
            "Policy Iteration": {"gamma": 0.99},
            "Value Iteration": {"gamma": 0.99},
            "Dyna Q": {"gamma": 0.95, "alpha": 0.1, "epsilon": 0.1, "planning_steps": 5},
            "Dyna Q+": {"gamma": 0.95, "alpha": 0.1, "epsilon": 0.1, "planning_steps": 5, "kappa": 0.001},
            "Sarsa": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1, "episodes": 100},
            "Expected Sarsa": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1, "episodes": 100},
            "Q Learning": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.3, "episodes": 10000},
            "First visit Monte Carlo": {"gamma": 0.9, "episodes": 500, "epsilon": 0.1},
            "Monte Carlo ES": {"gamma" : 0.9, "episodes": 1000},
            "Off-policy Monte Carlo": {"gamma": 0.9, "episodes": 1000}
        }

    def run(self):
        print(f"Bienvenue dans l'environnement Grid World avec l'agent '{self.agent_name}'")
        print("Souhaitez-vous :")
        print("1 - Charger une politique existante")
        print("2 - Apprendre une nouvelle politique")
        choix = input("Votre choix (1/2) : ").strip()

        filename = f"gridworld_{self.agent_name.replace(' ', '_').lower()}.pkl"

        if choix == "1":
            try:
                loaded_policy = load_policy(filename)
                # Si les clés sont des entiers, on fait le mapping index_to_state
                if isinstance(next(iter(loaded_policy)), int):
                    self.policy = {self.env.index_to_state[s]: a for s, a in loaded_policy.items()}
                else:
                    self.policy = loaded_policy
                print(f"Politique chargée depuis {filename}")
            except FileNotFoundError as e:
                print(e)
                return

        else:
            agent_func_map = {
                "Policy Iteration": policy_iteration,
                "Value Iteration": value_iteration,
                "Dyna Q": dyna_q,
                "Dyna Q+": dyna_q_plus,
                "Sarsa": sarsa,
                "Expected Sarsa": expected_sarsa,
                "First visit Monte Carlo": on_policy_first_visit_mc_control,
                "Monte Carlo ES": monte_carlo_es,
                "Off-policy Monte Carlo": off_policy_mc_control,
                "Q Learning": q_learning,
            }

            if self.agent_name not in agent_func_map:
                print(f"Agent '{self.agent_name}' non reconnu.")
                return

            try:
                agent_func = agent_func_map[self.agent_name]
                hyperparams = self.hyperparams_map.get(self.agent_name, {})
                policy_raw, Q = agent_func(self.env, **hyperparams)
                if isinstance(policy_raw, dict):
                    # Si les clés sont des entiers (ex: index d'état)
                    if isinstance(next(iter(policy_raw)), int):
                        self.policy = {self.env.index_to_state[s]: a for s, a in policy_raw.items()}
                    else:
                        self.policy = policy_raw
                else:
                    from agents.temporal_difference_methods import extract_deterministic_policy
                    raw_policy = extract_deterministic_policy(policy_raw)
                    self.policy = {self.env.index_to_state[s]: a for s, a in raw_policy.items()}


            except Exception as e:
                print(f"Erreur lors de l'exécution de l'agent {self.agent_name} : {e}")
                return

            print("Politique entraînée. Souhaitez-vous la sauvegarder ? (O/N)")
            if input().strip().lower() == "o":
                save_policy(self.policy, filename)

        print("Choisissez un mode :")
        print("1 - Agent automatique")
        print("2 - Joueur humain")
        mode = input("Mode : ").strip()

        hyperparams = self.hyperparams_map.get(self.agent_name, {})

        if mode == "1":
            self._run_agent(hyperparams)
        elif mode == "2":
            self._run_human()
        else:
            print("Mode invalide.")

    def _draw(self, policy, score, episode, message=""):
        self.screen.fill(WHITE)
        for row in range(self.env.height):
            for col in range(self.env.width):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                state = (row, col)
                if state in self.env.terminal_states:
                    pygame.draw.rect(self.screen, RED if self.env.get_reward(state) < 0 else GREEN, rect)
                else:
                    pygame.draw.rect(self.screen, GREY, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                if state in policy:
                    action = policy[state]
                    self._draw_arrow(row, col, action)

        agent_row, agent_col = self.env.agent_pos
        cx = agent_col * CELL_SIZE + CELL_SIZE // 2
        cy = agent_row * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, BLUE, (cx, cy), 15)

        self.screen.blit(self.font.render(f"Score: {score:.1f}", True, BLACK), (10, self.env.height * CELL_SIZE + 5))
        self.screen.blit(self.font.render(f"Épisode: {episode}", True, BLACK), (200, self.env.height * CELL_SIZE + 5))
        if message:
            self.screen.blit(self.font.render(message, True, BLACK), (400, self.env.height * CELL_SIZE + 5))
        pygame.display.flip()

    def _draw_arrow(self, row, col, action):
        cx = col * CELL_SIZE + CELL_SIZE // 2
        cy = row * CELL_SIZE + CELL_SIZE // 2
        offset = 20
        if action == 0:  # Haut
            pygame.draw.polygon(self.screen, BLACK, [(cx, cy - offset), (cx - 5, cy - 10), (cx + 5, cy - 10)])
        elif action == 1:  # Bas
            pygame.draw.polygon(self.screen, BLACK, [(cx, cy + offset), (cx - 5, cy + 10), (cx + 5, cy + 10)])
        elif action == 2:  # Gauche
            pygame.draw.polygon(self.screen, BLACK, [(cx - offset, cy), (cx - 10, cy - 5), (cx - 10, cy + 5)])
        elif action == 3:  # Droite
            pygame.draw.polygon(self.screen, BLACK, [(cx + offset, cy), (cx + 10, cy - 5), (cx + 10, cy + 5)])

    def _wait_for_restart(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

    def _run_agent(self, hyperparams):
        episode = 1
        while True:
            state = self.env.reset()
            total_reward = 0
            while True:
                self._draw(self.policy, total_reward, episode)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                pygame.time.delay(300)
                if self.env.is_terminal(state):
                    self._draw(self.policy, total_reward, episode, "Terminé - appuyez sur R")
                    break
                action = self.policy[state]
                next_state, reward = self.env.transition(state, action)
                total_reward += reward
                self.env.agent_pos = next_state
                state = next_state

            # Export des résultats
            export_results(
                agent_name=self.agent_name,
                env_name="GridWorld",
                stats={"score_total": total_reward, "nb_episodes": episode},
                hyperparams=hyperparams
            )

            episode += 1
            self._wait_for_restart()

    def _run_human(self):
        episode = 1
        while True:
            state = self.env.reset()
            total_reward = 0
            while True:
                self._draw({}, total_reward, episode)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if self.env.is_terminal(state):
                            break
                        if event.key == pygame.K_UP:
                            action = 0
                        elif event.key == pygame.K_DOWN:
                            action = 1
                        elif event.key == pygame.K_LEFT:
                            action = 2
                        elif event.key == pygame.K_RIGHT:
                            action = 3
                        else:
                            continue
                        next_state, reward = self.env.transition(state, action)
                        self.env.agent_pos = next_state
                        total_reward += reward
                        state = next_state
                if self.env.is_terminal(state):
                    break
            self._draw({}, total_reward, episode, "Terminé - appuyez sur R")
            episode += 1
            self._wait_for_restart()
