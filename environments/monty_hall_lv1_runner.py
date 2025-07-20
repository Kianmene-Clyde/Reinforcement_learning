import pygame
import sys
from Utils.save_load_policy import save_policy, load_policy
from Utils.export_results_to_xlsx import export_results
from agents.dynamic_programming import policy_iteration, value_iteration
from agents.planning_methods import dyna_q, dyna_q_plus
from agents.temporal_difference_methods import sarsa, expected_sarsa, q_learning
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control,
    monte_carlo_es,
    off_policy_mc_control
)
from environments.monty_hall_lv1_env import MontyHallEnv
import numpy as np

WHITE, BLACK, BLUE, GREEN, RED, GREY = (
    (255, 255, 255), (0, 0, 0), (50, 50, 255), (50, 200, 50), (255, 80, 80), (200, 200, 200)
)
WIDTH, HEIGHT = 600, 300
CELL_WIDTH = 100


class MontyHallRunner:
    def __init__(self, agent_name="Policy Iteration"):
        pygame.init()
        self.env = MontyHallEnv()
        self.agent_name = agent_name
        self.policy = {}
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption(f"Monty Hall - {self.agent_name}")

        self.hyperparams_map = {
            "Policy Iteration": {"gamma": 0.99},
            "Value Iteration": {"gamma": 0.99},
            "Dyna Q": {"gamma": 0.95, "alpha": 0.1, "epsilon": 0.1, "planning_steps": 5},
            "Dyna Q+": {"gamma": 0.95, "alpha": 0.1, "epsilon": 0.1, "planning_steps": 5, "kappa": 0.001},
            "Sarsa": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1, "episodes": 100},
            "Expected Sarsa": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1, "episodes": 100},
            "Q Learning": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.3, "episodes": 10000},
            "First visit Monte Carlo": {"gamma": 0.9, "episodes": 500, "epsilon": 0.1},
            "Monte Carlo ES": {"episodes": 1000},
            "Off-policy Monte Carlo": {"gamma": 0.9, "episodes": 500}
        }

    def run(self):
        print(f"Bienvenue dans Monty Hall LV1 avec l'agent '{self.agent_name}'")
        print("Souhaitez-vous :")
        print("1 - Charger une politique existante")
        print("2 - Apprendre une nouvelle politique")
        choix = input("Votre choix (1/2) : ").strip()

        filename = f"montyhall_lv1_{self.agent_name.replace(' ', '_').lower()}.pkl"

        if choix == "1":
            try:
                self.policy = load_policy(filename)
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
                self.policy, _ = agent_func(self.env, **hyperparams)
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

    def _draw(self, state, reward=None, message="", action_str=""):
        self.screen.fill(WHITE)
        label = self.font.render(message, True, BLACK)
        self.screen.blit(label, (20, 20))
        if action_str:
            action_label = self.font.render(f"Action: {action_str}", True, BLUE)
            self.screen.blit(action_label, (WIDTH - 200, 20))

        for i in range(3):
            color = GREY
            label_txt = ""

            if isinstance(state, tuple):
                phase = state[0]
                chosen = state[1] if len(state) > 1 else None
                opened = state[2] if len(state) > 2 else None

                if phase == "start":
                    if i == chosen:
                        color = BLUE
                        label_txt = "Choisie"
                elif phase == "reveal":
                    if i == opened:
                        color = RED
                        label_txt = "Ouverte"
                    elif i == chosen:
                        color = BLUE
                        label_txt = "Choisie"
                elif phase == "done":
                    if i == self.env.winning_door:
                        color = GREEN
                        label_txt = "Gagnante"
                    elif i == chosen:
                        color = BLUE
                        label_txt = "Choisie"

            pygame.draw.rect(self.screen, color, (100 + i * 150, 100, CELL_WIDTH, 150))
            num = self.font.render(str(i), True, BLACK)
            self.screen.blit(num, (140 + i * 150, 110))

            if label_txt:
                tag = self.font.render(label_txt, True, BLACK)
                self.screen.blit(tag, (110 + i * 150, 220))

        if isinstance(state, tuple) and state[0] == "done":
            result = "GAGNÉ !" if reward == 1.0 else "PERDU"
            label2 = self.font.render(f"{result} | R pour rejouer", True, BLACK)
            self.screen.blit(label2, (20, 60))

        pygame.display.flip()

    def _wait_for_restart(self):
        while True:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            pygame.time.delay(100)

    def _run_agent(self, hyperparams):
        episode = 1
        while True:
            state = self.env.reset()
            reward = 0.0
            while not self.env.is_terminal(state):
                state_index = self.env.state_to_index[state]
                action = np.argmax(self.policy[state_index])
                action_str = "GARDER" if action == 0 else "CHANGER"
                print(f"[Agent] État: {state} -> Action: {action_str}")
                self._draw(state, reward, f"Épisode {episode}", action_str)
                pygame.time.delay(1000)
                next_state, reward = self.env.step(action)
                state = next_state
            self._draw(state, reward, f"Terminé | Épisode {episode}")

            export_results(
                agent_name=self.agent_name,
                env_name="MontyHall LV1",
                stats={"score_total": reward, "nb_episodes": episode},
                hyperparams=hyperparams
            )

            episode += 1
            self._wait_for_restart()

    def _run_human(self):
        episode = 1
        while True:
            state = self.env.reset()
            state = ("start", None, None)
            reward = 0.0
            while not self.env.is_terminal(state):
                self._draw(state, reward, f"Épisode {episode} - Choisissez une action")
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        actions = self.env.get_actions(state)
                        if event.unicode.isdigit():
                            a = int(event.unicode)
                            if a in actions:
                                next_state, reward = self.env.step(a)
                                state = next_state
                pygame.time.delay(200)
            self._draw(state, reward, f"Terminé | Épisode {episode}")
            episode += 1
            self._wait_for_restart()
