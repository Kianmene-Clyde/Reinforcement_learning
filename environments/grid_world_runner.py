import pygame
import sys
from agents.dynamic_programming import policy_iteration, value_iteration
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

    def run(self):
        print("Choisissez un mode :")
        print("1 - Agent automatique")
        print("2 - Joueur humain")
        mode = input("Mode : ").strip()

        if self.agent_name == "Policy Iteration":
            self.policy, _ = policy_iteration(self.env, gamma=0.99)

        elif self.agent_name == "Value Iteration":
            self.policy, _ = value_iteration(self.env)

        if mode == "1":
            self._run_agent()
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
                    pygame.quit();
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit();
                        sys.exit()

    def _run_agent(self):
        episode = 1
        while True:
            state = self.env.reset()
            total_reward = 0
            while True:
                self._draw(self.policy, total_reward, episode)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit();
                        sys.exit()

                pygame.time.delay(300)
                if self.env.is_terminal(state):
                    self._draw(self.policy, total_reward, episode, "✅ Terminé - appuyez sur R")
                    break
                action = self.policy[state]
                next_state, reward = self.env.transition(state, action)
                total_reward += reward
                self.env.agent_pos = next_state
                state = next_state
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
                        pygame.quit();
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
            self._draw({}, total_reward, episode, "✅ Terminé - appuyez sur R")
            episode += 1
            self._wait_for_restart()
