import pygame
import sys
from agents.dynamic_programming import policy_iteration, value_iteration
from agents.planning_methods import dyna_q, dyna_q_plus
from agents.temporal_difference_methods import sarsa, expected_sarsa, q_learning
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control,
    monte_carlo_es,
    off_policy_mc_control
)
from environments.monty_hall_lv2_env import MontyHallEnvLv2
from Utils.save_load_policy import save_policy, load_policy

WHITE, BLACK, GREEN, RED, GREY = (255, 255, 255), (0, 0, 0), (50, 200, 50), (255, 80, 80), (200, 200, 200)
WIDTH, HEIGHT = 800, 400
CELL_WIDTH = 100


class MontyHallRunnerLv2:
    def __init__(self, agent_name="Policy Iteration"):
        pygame.init()
        self.env = MontyHallEnvLv2()
        self.agent_name = agent_name
        self.policy = {}
        self.results = []
        self.hyperparams = {}
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("Arial", 22)
        pygame.display.set_caption(f"Monty Hall LV2 - {self.agent_name}")

    def run(self):
        print(f"Bienvenue dans Monty Hall LV2 avec l'agent '{self.agent_name}'")
        print("Souhaitez-vous :")
        print("1 - Charger une politique existante")
        print("2 - Apprendre une nouvelle politique")
        choix = input("Votre choix (1/2) : ").strip()

        filename = f"montyhall_lv2_{self.agent_name.replace(' ', '_').lower()}.pkl"

        if choix == "1":
            try:
                self.policy = load_policy(filename)
                self.hyperparams = {}  # Valeur vide si on ne les connaît pas
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
                print(f"⚠️ Agent '{self.agent_name}' non reconnu.")
                return

            try:
                agent_func = agent_func_map[self.agent_name]
                self.policy, self.hyperparams = agent_func(self.env)
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

        if mode == "1":
            self._run_agent()
        elif mode == "2":
            self._run_human()
        else:
            print("Mode invalide.")

    def _draw(self, state, reward=None, message=""):
        self.screen.fill(WHITE)
        self.screen.blit(self.font.render(message, True, BLACK), (20, 20))

        if isinstance(state, tuple) and state[0] == "reveal":
            chosen = state[1]
            revealed = state[2]
            possible = [d for d in self.env.doors if d not in revealed]
            self.screen.blit(self.font.render(f"Choix finaux possibles : {possible}", True, BLACK), (20, 60))
            for i in range(5):
                color = RED if i in revealed else GREEN if i == chosen else GREY
                pygame.draw.rect(self.screen, color, (60 + i * 140, 100, CELL_WIDTH, 150))
                self.screen.blit(self.font.render(str(i), True, BLACK), (100 + i * 140, 160))

        elif isinstance(state, tuple) and state[0] == "done":
            final_choice = state[1]
            for i in range(5):
                color = GREEN if i == self.env.winning_door else GREY
                pygame.draw.rect(self.screen, color, (60 + i * 140, 100, CELL_WIDTH, 150))
                self.screen.blit(self.font.render(str(i), True, BLACK), (100 + i * 140, 160))
            result = "✅ GAGNÉ !" if reward == 1.0 else "❌ PERDU"
            label2 = self.font.render(f"{result} | R pour rejouer | Échap pour quitter", True, BLACK)
            self.screen.blit(label2, (20, 60))

        elif isinstance(state, tuple) and state[0] == "start":
            for i in range(5):
                pygame.draw.rect(self.screen, GREY, (60 + i * 140, 100, CELL_WIDTH, 150))
                self.screen.blit(self.font.render(str(i), True, BLACK), (100 + i * 140, 160))

        pygame.display.flip()

    def _wait_for_restart(self):
        while True:
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

    def _run_agent(self):
        episode = 1
        while True:
            state = self.env.reset()
            reward = 0.0
            while not self.env.is_terminal(state):
                self._draw(state, reward, f"Épisode {episode}")
                pygame.time.delay(1000)
                action = self.policy.get(state, 0)
                state, reward = self.env.step(state, action)
            self._draw(state, reward, f"✅ Terminé | Épisode {episode}")
            self.results.append(reward)
            export_results(
                agent_name=self.agent_name,
                env_name="MontyHall LV2",
                stats={"Score": reward, "Épisode": episode},
                hyperparams=self.hyperparams,
                filename="montyhall_lv2_results.xlsx"
            )
            episode += 1
            self._wait_for_restart()

    def _run_human(self):
        episode = 1
        while True:
            state = self.env.reset()
            reward = 0.0
            while not self.env.is_terminal(state):
                self._draw(state, reward, f"Épisode {episode} - Choisissez une action")
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            sys.exit()
                        actions = self.env.get_actions(state)
                        if event.unicode.isdigit():
                            a = int(event.unicode)
                            if a in actions:
                                state, reward = self.env.step(state, a)
                pygame.time.delay(200)
            self._draw(state, reward, f"✅ Terminé | Épisode {episode}")
            episode += 1
            self._wait_for_restart()
