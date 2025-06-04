import pygame
import sys
from environments.grid_world_runner import GridWorldRunner
from environments.line_world_runner import LineWorldRunner
from environments.monty_hall_lv1_runner import MontyHallRunner
from environments.monty_hall_lv2_runner import MontyHallRunnerLv2

pygame.init()
WIDTH, HEIGHT = 600, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Menu Graphique")
font = pygame.font.SysFont("Arial", 26)
small_font = pygame.font.SysFont("Arial", 20)
WHITE, BLACK, GREY, GREEN = (255, 255, 255), (0, 0, 0), (200, 200, 200), (50, 200, 50)

AGENTS = [
    "Policy Iteration", "Value Iteration", "Dyna Q", "Dyna Q+", "Sarsa",
    "Expected Sarsa", "First visit Monte Carlo", "Monte Carlo ES",
    "Off-policy Monte Carlo", "Q Learning"
]

ENVS = [
    "LineWorld", "GridWorld", "Monty Hall lvl 1", "Monty Hall lvl 2", "RPS"
]

selected_agent = 0
selected_env = 0
step = 0  # 0 = agent, 1 = env

running = True
while running:
    screen.fill(WHITE)
    title = font.render("Choisissez un agent" if step == 0 else "Choisissez un environnement", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))

    options = AGENTS if step == 0 else ENVS
    for i, name in enumerate(options):
        color = GREEN if (step == 0 and i == selected_agent) or (step == 1 and i == selected_env) else GREY
        pygame.draw.rect(screen, color, (100, 70 + i * 50, 400, 40), border_radius=8)
        label = small_font.render(f"{i + 1} - {name}", True, BLACK)
        screen.blit(label, (110, 80 + i * 50))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_9:
                index = event.key - pygame.K_1
            elif event.key == pygame.K_0:
                index = 9
            else:
                continue
            if step == 0 and index < len(AGENTS):
                selected_agent = index
                step = 1
            elif step == 1 and index < len(ENVS):
                selected_env = index
                running = False

pygame.quit()

# === Lancement logique ===
agent_name = AGENTS[selected_agent]
if selected_env == 0:
    if agent_name == "Policy Iteration":
        LineWorldRunner(agent_name=agent_name).run()
    else:
        print("Agent non encore supporté pour LineWorld.")

elif selected_env == 1:
    if agent_name == "Policy Iteration":
        GridWorldRunner(agent_name=agent_name).run()
    else:
        print("Agent non encore supporté pour GridWorld.")

elif selected_env == 2:
    if agent_name == "Policy Iteration":
        MontyHallRunner(agent_name=agent_name).run()
    else:
        print("Agent non encore supporté pour Monty Hall lvl 1.")

elif selected_env == 3:
    if agent_name == "Policy Iteration":
        MontyHallRunnerLv2(agent_name=agent_name).run()
    else:
        print("Agent non encore supporté pour Monty Hall lvl 2.")

else:
    print("Environnement non reconnu.")
