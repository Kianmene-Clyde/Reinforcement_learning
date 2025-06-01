import pygame
import sys
import time


# === Définition de l’environnement Line World ===
class LineWorldEnv:
    def __init__(self, length=5):
        self.length = length
        self.states = list(range(length))
        self.terminal_states = [0, length - 1]
        self.agent_pos = 2

    def reset(self, pos=2):
        self.agent_pos = pos
        return self.agent_pos

    def get_actions(self, state):
        return [0, 1] if state not in self.terminal_states else []

    def transition(self, state, action):
        if state in self.terminal_states:
            return state, 0.0
        next_state = max(0, state - 1) if action == 0 else min(self.length - 1, state + 1)
        reward = -1.0 if next_state == 0 else 1.0 if next_state == self.length - 1 else 0.0
        return next_state, reward


# === Politique fixe ===
policy = {s: 1 if s < 4 else 0 for s in range(5)}

# === Paramètres d'affichage ===
CELL_SIZE = 100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
GREY = (180, 180, 180)


# Dessin d'une flèche
def draw_arrow(surface, x, y, direction):
    if direction == 1:  # Right
        pygame.draw.polygon(surface, BLACK, [(x, y), (x - 10, y - 10), (x - 10, y + 10)])
    else:  # Left
        pygame.draw.polygon(surface, BLACK, [(x, y), (x + 10, y - 10), (x + 10, y + 10)])


# Affichage global
def draw(env, screen, score, episode, message=""):
    screen.fill(WHITE)
    for i in env.states:
        rect = pygame.Rect(i * CELL_SIZE + 10, 100, CELL_SIZE - 20, 100)
        color = RED if i == 0 else GREEN if i == env.length - 1 else GREY
        pygame.draw.rect(screen, color, rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, rect, 2)
        label = font.render(str(i), True, BLACK)
        screen.blit(label, (i * CELL_SIZE + CELL_SIZE // 2 - 5, 110))

        if i not in env.terminal_states:
            direction = policy.get(i)
            if direction is not None:
                draw_arrow(screen, i * CELL_SIZE + CELL_SIZE // 2, 210, direction)

    # Agent
    x = env.agent_pos * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, (0, 0, 255), (x, 170), 20)

    # Score et épisode
    score_label = font.render(f"Score: {score:.1f}", True, BLACK)
    ep_label = font.render(f"Épisode: {episode}", True, BLACK)
    screen.blit(score_label, (10, 20))
    screen.blit(ep_label, (200, 20))

    if message:
        end_label = font.render(message, True, BLACK)
        screen.blit(end_label, (10, 60))

    pygame.display.flip()


# Attente touche R ou Esc
def wait_for_restart():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit();
                    sys.exit()


# Mode Agent
def run_agent(env):
    episode = 1
    while True:
        state = env.reset()
        total_reward = 0
        while True:
            draw(env, screen, total_reward, episode)
            pygame.time.delay(600)
            if state in env.terminal_states:
                break
            action = policy[state]
            next_state, reward = env.transition(state, action)
            print(f"Agent: s={state}, a={'Left' if action == 0 else 'Right'} -> s'={next_state}, r={reward}")
            state = next_state
            total_reward += reward
            env.agent_pos = state

        draw(env, screen, total_reward, episode, "✅ Terminé - appuyez sur R pour rejouer")
        episode += 1
        wait_for_restart()


# Mode Joueur Humain
def run_human(env):
    episode = 1
    while True:
        state = env.reset()
        total_reward = 0
        while True:
            draw(env, screen, total_reward, episode)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if state in env.terminal_states:
                        break
                    if event.key == pygame.K_LEFT:
                        next_state, reward = env.transition(state, 0)
                        print(f"You pressed LEFT: s={state} -> s'={next_state}, r={reward}")
                        state = next_state
                    elif event.key == pygame.K_RIGHT:
                        next_state, reward = env.transition(state, 1)
                        print(f"You pressed RIGHT: s={state} -> s'={next_state}, r={reward}")
                        state = next_state
                    total_reward += reward
                    env.agent_pos = state
            if state in env.terminal_states:
                break

        draw(env, screen, total_reward, episode, "✅ Terminé - appuyez sur R pour rejouer")
        episode += 1
        wait_for_restart()


# === Lancement PyGame ===
pygame.init()
font = pygame.font.SysFont("Arial", 24)
screen = pygame.display.set_mode((5 * CELL_SIZE, 300))
pygame.display.set_caption("Line World - RL Visualisation")

env = LineWorldEnv()
print("Choisissez un mode :")
print("1 - Mode agent automatique (policy apprise)")
print("2 - Mode joueur humain (touches ← et →)")
mode = input("Entrée (1 ou 2) : ").strip()

if mode == "1":
    run_agent(env)
elif mode == "2":
    run_human(env)
else:
    print("Mode invalide.")
