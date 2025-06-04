import pygame
import sys
import time
from agents.dynamic_programming import policy_iteration
from environments.rps_game_env import RPSGameEnv

WHITE, BLACK, GREEN, RED, BLUE, GREY = (255, 255, 255), (0, 0, 0), (50, 200, 50), (255, 80, 80), (50, 50, 255), (200,
                                                                                                                 200,
                                                                                                                 200)
WIDTH, HEIGHT = 800, 400
CELL_WIDTH = 150
CHOICES = ["Pierre", "Feuille", "Ciseaux"]


class RPSGameRunner:
    def __init__(self, agent_name="Policy Iteration"):
        pygame.init()
        self.env = RPSGameEnv()
        self.agent_name = agent_name
        self.policy, _ = policy_iteration(self.env, gamma=0.99)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption(f"RPS Game - {self.agent_name}")

    def run(self):
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

    def _draw(self, message, choice_a=None, choice_b=None, result=None):
        self.screen.fill(WHITE)
        label = self.font.render(message, True, BLACK)
        self.screen.blit(label, (20, 20))

        for i, choice in enumerate(CHOICES):
            pygame.draw.rect(self.screen, GREY, (50 + i * (CELL_WIDTH + 30), 100, CELL_WIDTH, 100))
            text = self.font.render(f"{i} - {choice}", True, BLACK)
            self.screen.blit(text, (60 + i * (CELL_WIDTH + 30), 130))

        if choice_a is not None and choice_b is not None:
            result_text = "Egalite" if result == 0 else "Gagne" if result == 1 else "Perdu"
            color = GREEN if result == 1 else RED if result == -1 else BLUE
            outcome = self.font.render(f"Agent: {CHOICES[choice_a]} vs Ennemi: {CHOICES[choice_b]} => {result_text}",
                                       True, color)
            self.screen.blit(outcome, (20, 250))

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
                self._draw(f"Episode {episode} - Agent joue", *state)
                pygame.time.delay(1000)
                action = self.policy[state]
                print("[Agent] Etat:", state, "-> Action:", action)
                state, reward = self.env.step(state, action)
            self._draw(f"Episode {episode} - Terminé", *state, reward)
            episode += 1
            self._wait_for_restart()

    def _run_human(self):
        episode = 1
        while True:
            state = self.env.reset()
            reward = 0.0
            while not self.env.is_terminal(state):
                self._draw(f"Episode {episode} - Choisissez une action")
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit();
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.unicode.isdigit():
                            a = int(event.unicode)
                            if a in [0, 1, 2]:
                                state, reward = self.env.step(state, a)
                                break
                pygame.time.delay(100)
            self._draw(f"Episode {episode} - Terminé", *state, reward)
            episode += 1
            self._wait_for_restart()
