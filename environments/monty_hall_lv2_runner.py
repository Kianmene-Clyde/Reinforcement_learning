import pygame
import sys
import time
from agents.dynamic_programming import policy_iteration
from environments.monty_hall_lv2_env import MontyHallEnvLv2

WHITE, BLACK, GREEN, RED, GREY = (255, 255, 255), (0, 0, 0), (50, 200, 50), (255, 80, 80), (200, 200, 200)
WIDTH, HEIGHT = 800, 400
CELL_WIDTH = 100


class MontyHallRunnerLv2:
    def __init__(self, agent_name="Policy Iteration"):
        pygame.init()
        self.env = MontyHallEnvLv2()
        self.agent_name = agent_name
        self.policy = {}
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("Arial", 22)
        pygame.display.set_caption(f"Monty Hall LV2 - {self.agent_name}")

    def run(self):
        print("Choisissez un mode :")
        print("1 - Agent automatique")
        print("2 - Joueur humain")
        mode = input("Mode : ").strip()

        if self.agent_name == "Policy Iteration":
            self.policy, _ = policy_iteration(self.env, gamma=0.99)
            print("Politique apprise :", self.policy)

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
                    pygame.quit();
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit();
                        sys.exit()

    def _run_agent(self):
        episode = 1
        while True:
            state = self.env.reset()
            reward = 0.0
            while not self.env.is_terminal(state):
                self._draw(state, reward, f"Épisode {episode}")
                pygame.time.delay(1000)
                action = self.policy[state]
                state, reward = self.env.step(state, action)
            self._draw(state, reward, f"✅ Terminé | Épisode {episode}")
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
                        pygame.quit();
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit();
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
