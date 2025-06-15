import pygame
import sys
from environments.grid_world_runner import GridWorldRunner
from environments.line_world_runner import LineWorldRunner
from environments.monty_hall_lv1_runner import MontyHallRunner
from environments.monty_hall_lv2_runner import MontyHallRunnerLv2

pygame.init()
WIDTH, HEIGHT = 700, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Menu Graphique Avancé")
font = pygame.font.SysFont("Arial", 30)
small_font = pygame.font.SysFont("Arial", 22)
WHITE, BLACK, GREY, GREEN = (255, 255, 255), (0, 0, 0), (210, 210, 210), (50, 200, 50)

AGENTS = [
    "Policy Iteration", "Value Iteration", "Dyna Q", "Dyna Q+", "Sarsa",
    "Expected Sarsa", "First visit Monte Carlo", "Monte Carlo ES",
    "Off-policy Monte Carlo", "Q Learning"
]

ENVS = ["LineWorld", "GridWorld", "Monty Hall lvl 1", "Monty Hall lvl 2"]

selected_agent = 0
selected_env = 0
step = 0  # 0 = agent, 1 = env
running = True

bg_image = pygame.Surface((WIDTH, HEIGHT))
bg_image.fill(WHITE)
pygame.draw.circle(bg_image, (240, 240, 255), (WIDTH // 2, HEIGHT // 2), 300)

while running:
    screen.blit(bg_image, (0, 0))
    title = font.render("Choisissez un agent" if step == 0 else "Choisissez un environnement", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

    options = AGENTS if step == 0 else ENVS
    for i, name in enumerate(options):
        color = GREEN if (step == 0 and i == selected_agent) or (step == 1 and i == selected_env) else GREY
        pygame.draw.rect(screen, color, (80, 80 + i * 45, 540, 40), border_radius=10)
        label = small_font.render(f"{i + 1} - {name}", True, BLACK)
        screen.blit(label, (100, 90 + i * 45))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if step == 0:
                if event.unicode.isdigit():
                    index = int(event.unicode) - 1
                    if 0 <= index < len(AGENTS):
                        selected_agent = index
                        step = 1
                elif event.unicode.lower() == 'q':
                    selected_agent = 9
                    step = 1

            elif step == 1:
                if event.unicode.isdigit():
                    index = int(event.unicode) - 1
                    if 0 <= index < len(ENVS):
                        selected_env = index
                        running = False

# === Lancement logique ===
agent_name = AGENTS[selected_agent]
env_id = selected_env

pygame.display.quit()

if env_id == 0:
    LineWorldRunner(agent_name=agent_name).run()
elif env_id == 1:
    GridWorldRunner(agent_name=agent_name).run()
elif env_id == 2:
    MontyHallRunner(agent_name=agent_name).run()
elif env_id == 3:
    MontyHallRunnerLv2(agent_name=agent_name).run()
else:
    print("Environnement non reconnu.")
