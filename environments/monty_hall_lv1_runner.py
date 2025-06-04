import pygame
import sys
import time
from agents.dynamic_programming import policy_iteration
from environments.monty_hall_lv1_env import MontyHallEnv

WHITE, BLACK, BLUE, GREEN, RED, GREY = (255, 255, 255), (0, 0, 0), (50, 50, 255), (50, 200, 50), (255, 80, 80), (200,
                                                                                                                 200,
                                                                                                                 200)
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

    def _draw(self, state, reward=None, message="", action_str=""):
        self.screen.fill(WHITE)
        label = self.font.render(message, True, BLACK)
        self.screen.blit(label, (20, 20))
        if action_str:
            action_label = self.font.render(f"Action: {action_str}", True, BLUE)
            self.screen.blit(action_label, (WIDTH - 200, 20))

        if isinstance(state, tuple) and state[0] == "reveal":
            chosen = state[1]
            revealed = state[2]
            for i in range(3):
                color = RED if i == revealed else GREEN if i == chosen else GREY
                pygame.draw.rect(self.screen, color, (100 + i * 150, 100, CELL_WIDTH, 150))
                num = self.font.render(str(i), True, BLACK)
                self.screen.blit(num, (140 + i * 150, 160))

        elif isinstance(state, tuple) and state[0] == "done":
            final_choice = state[1]
            for i in range(3):
                color = GREEN if i == self.env.winning_door else GREY
                pygame.draw.rect(self.screen, color, (100 + i * 150, 100, CELL_WIDTH, 150))
                num = self.font.render(str(i), True, BLACK)
                self.screen.blit(num, (140 + i * 150, 160))
            result = "✅ GAGNÉ !" if reward == 1.0 else "❌ PERDU"
            label2 = self.font.render(f"{result} | R pour rejouer", True, BLACK)
            self.screen.blit(label2, (20, 60))

        elif state == "start" or (isinstance(state, tuple) and state[0] == "start"):
            for i in range(3):
                pygame.draw.rect(self.screen, GREY, (100 + i * 150, 100, CELL_WIDTH, 150))
                num = self.font.render(str(i), True, BLACK)
                self.screen.blit(num, (140 + i * 150, 160))

        pygame.display.flip()

    def _wait_for_restart(self):
        while True:
            pygame.event.pump()  # ✅ force la mise à jour des événements
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

    def _run_agent(self):
        episode = 1
        while True:
            state = self.env.reset()
            reward = 0.0
            while not self.env.is_terminal(state):
                action = self.policy.get(state, 0)
                action_str = "GARDER" if action == 0 else "CHANGER"
                print(f"[Agent] État: {state} -> Action: {action_str}")
                self._draw(state, reward, f"Épisode {episode}", action_str)
                pygame.time.delay(1000)
                next_state, reward = self.env.step(state, action)
                state = next_state
            self._draw(state, reward, f"✅ Terminé | Épisode {episode}")
            episode += 1
            self._wait_for_restart()

    def _run_human(self):
        episode = 1
        while True:
            state = self.env.reset()
            reward = 0.0
            while not self.env.is_terminal(state):
                self._draw(state, reward, f"Épisode {episode} - Choisissez une action (0, 1, 2)")
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        actions = self.env.get_actions(state)
                        if event.unicode.isdigit():
                            a = int(event.unicode)
                            if a in actions:
                                next_state, reward = self.env.step(state, a)
                                state = next_state
                pygame.time.delay(200)
            self._draw(state, reward, f"✅ Terminé | Épisode {episode}")
            episode += 1
            self._wait_for_restart()
