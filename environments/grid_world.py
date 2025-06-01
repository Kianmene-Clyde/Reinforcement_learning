import pygame
import sys
import numpy as np


# === ENVIRONNEMENT GRIDWORLD ===
class GridWorldEnv:
    def __init__(self, rows=10, cols=10, start=(0, 0), goal=(9, 9), hole=(7, 7), obstacles=None):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.hole = hole
        self.obstacles = set(obstacles) if obstacles else set()
        self.terminal_states = [goal, hole]
        self.actions = {
            0: (-1, 0),  # Haut
            1: (1, 0),  # Bas
            2: (0, -1),  # Gauche
            3: (0, 1)  # Droite
        }
        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def get_actions(self, state):
        return list(self.actions.keys()) if state not in self.terminal_states else []

    def transition(self, state, action):
        if state in self.terminal_states:
            return state, 0.0
        dr, dc = self.actions[action]
        r, c = state
        new_r = max(0, min(self.rows - 1, r + dr))
        new_c = max(0, min(self.cols - 1, c + dc))
        next_state = (new_r, new_c)
        if next_state in self.obstacles:
            next_state = state  # obstacle = pas de déplacement
        reward = 1.0 if next_state == self.goal else -1.0 if next_state == self.hole else 0.0
        return next_state, reward


# === AFFICHAGE PYGAME ===
CELL_SIZE = 60
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
WHITE, BLACK, RED, GREEN, GREY, BLUE, ORANGE = (255, 255, 255), (0, 0, 0), (200, 50, 50), (50, 200, 50), (200, 200,
                                                                                                          200), (50, 50,
                                                                                                                 255), (
    150, 100, 50)


# Option : une politique si elle est définie (None = ne pas afficher de flèche)
def default_policy(state):
    r, c = state
    if r < 9:
        return 1  # ↓
    elif c < 9:
        return 3  # →
    return None  # rien


def draw(env, screen, font, score, episode, show_policy=True, message=""):
    screen.fill(WHITE)
    for r in range(env.rows):
        for c in range(env.cols):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            color = ORANGE if (r, c) in env.obstacles else RED if (r, c) == env.hole else GREEN if (r,
                                                                                                    c) == env.goal else GREY
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)

            if show_policy and (r, c) not in env.terminal_states and (r, c) not in env.obstacles:
                a = default_policy((r, c))
                if a is not None:
                    label = font.render(ARROWS[a], True, BLACK)
                    screen.blit(label, (x + CELL_SIZE // 2 - 8, y + CELL_SIZE // 2 - 12))

    # Agent
    x, y = env.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2, env.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, BLUE, (x, y), 18)

    # Score & épisode
    label = font.render(f"Score: {score:.1f}  Épisode: {episode}", True, BLACK)
    screen.blit(label, (10, env.rows * CELL_SIZE + 5))

    if message:
        msg = font.render(message, True, BLACK)
        screen.blit(msg, (10, env.rows * CELL_SIZE + 30))

    pygame.display.flip()


def wait_for_restart():
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


def run_agent(env, screen, font):
    episode = 1
    while True:
        state = env.reset()
        score = 0
        while True:
            draw(env, screen, font, score, episode)
            pygame.time.delay(400)
            if state in env.terminal_states:
                draw(env, screen, font, score, episode, message="✅ Terminé - R pour rejouer")
                episode += 1
                wait_for_restart()
                break
            a = default_policy(state)
            next_state, r = env.transition(state, a)
            env.agent_pos = next_state
            state = next_state
            score += r


def run_human(env, screen, font):
    episode = 1
    while True:
        state = env.reset()
        score = 0
        while True:
            draw(env, screen, font, score, episode)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if state in env.terminal_states:
                        draw(env, screen, font, score, episode, message="✅ Terminé - R pour rejouer")
                        episode += 1
                        wait_for_restart()
                        state = env.reset()
                        score = 0
                        break
                    key_map = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
                    if event.key in key_map:
                        a = key_map[event.key]
                        next_state, r = env.transition(state, a)
                        env.agent_pos = next_state
                        state = next_state
                        score += r


# === MAIN ===
pygame.init()
font = pygame.font.SysFont("Arial", 20)
screen = pygame.display.set_mode((10 * CELL_SIZE, 10 * CELL_SIZE + 60))
pygame.display.set_caption("GridWorld 10x10")

obstacles = [(1, 1), (1, 2), (2, 2), (3, 1), (5, 5), (6, 5), (6, 6), (7, 5)]
env = GridWorldEnv(obstacles=obstacles)

print("1 - Agent automatique")
print("2 - Joueur humain (←↑→↓)")
mode = input("Choix : ").strip()

if mode == "1":
    run_agent(env, screen, font)
elif mode == "2":
    run_human(env, screen, font)
else:
    print("Choix invalide.")
