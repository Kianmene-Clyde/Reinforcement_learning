import pygame
import sys
import random

from agents.dynamic_programming import policy_iteration, value_iteration
from agents.monte_carlo_methods import monte_carlo_es, off_policy_mc_control, on_policy_first_visit_mc_control
from agents.temporal_difference_methods import sarsa, expected_sarsa, q_learning
from agents.planning_methods import dyna_q, dyna_q_plus
from environments.rps_game_env import RPSGameEnv

WIDTH, HEIGHT = 700, 400
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, BLUE = (0, 180, 0), (180, 0, 0), (0, 0, 180)
GREY = (230, 230, 230)
CHOICES = ["Pierre", "Feuille", "Ciseaux"]
WIN_MAP = {0: 2, 1: 0, 2: 1}

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Paper Scissors - Score Cumulé")
font = pygame.font.SysFont("Arial", 24)


def draw_message(text):
    screen.fill(WHITE)
    label = font.render(text, True, BLACK)
    screen.blit(label, (WIDTH // 2 - label.get_width() // 2, 40))
    pygame.display.flip()


def draw_choices():
    screen.fill(WHITE)
    for i in range(3):
        pygame.draw.rect(screen, GREY, (100 + i * 180, 150, 120, 100))
        pygame.draw.rect(screen, BLACK, (100 + i * 180, 150, 120, 100), 2)
        label = font.render(f"{i} = {CHOICES[i]}", True, BLACK)
        screen.blit(label, (110 + i * 180, 190))
    pygame.display.flip()


def draw_result(player, enemy, result, round_num, stats, is_agent=False):
    screen.fill(WHITE)
    result_text = "ÉGALITÉ" if result == 0 else "✅ GAGNÉ !" if result == 1 else "❌ PERDU"
    who = "Agent" if is_agent else "Vous"
    text = f"Round {round_num} - {who}: {CHOICES[player]} vs Ennemi: {CHOICES[enemy]} ➜ {result_text}"
    color = GREEN if result == 1 else RED if result == -1 else BLACK
    label = font.render(text, True, color)
    screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - 30))
    draw_score(stats)
    pygame.display.flip()


def draw_score(stats):
    win, loss, draw = stats["win"], stats["loss"], stats["draw"]
    screen.blit(font.render(f"✅ Victoires : {win}", True, GREEN), (20, HEIGHT - 60))
    screen.blit(font.render(f"❌ Défaites : {loss}", True, RED), (250, HEIGHT - 60))
    screen.blit(font.render(f"🤝 Égalités : {draw}", True, BLUE), (480, HEIGHT - 60))


def wait_for_key():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                return
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


def show_agent_menu():
    agents = [
        "Policy Iteration", "Value Iteration", "Dyna Q", "Dyna Q+",
        "Sarsa", "Expected Sarsa", "First Visit MC", "MC ES",
        "Off-policy MC", "Q Learning", "Joueur Humain"
    ]
    screen.fill(WHITE)
    draw_message("Sélectionnez un agent :")
    for i, text in enumerate(agents):
        key_label = f"{i + 1}" if i < 10 else "A"
        pygame.draw.rect(screen, GREY, (150, 70 + i * 30, 400, 30))
        label = font.render(f"{key_label} - {text}", True, BLACK)
        screen.blit(label, (160, 75 + i * 30))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                key = event.unicode
                if key.isdigit():
                    index = int(key) - 1 if key != '0' else 9
                    if 0 <= index < len(agents):
                        return agents[index]
                elif key.lower() == 'a':
                    return agents[10]


def rps_game(agent_name="Joueur Humain"):
    env = RPSGameEnv()
    agent_funcs = {
        "Policy Iteration": policy_iteration,
        "Value Iteration": value_iteration,
        "Dyna Q": dyna_q,
        "Dyna Q+": dyna_q_plus,
        "Sarsa": sarsa,
        "Expected Sarsa": expected_sarsa,
        "First Visit MC": on_policy_first_visit_mc_control,
        "MC ES": monte_carlo_es,
        "Off-policy MC": off_policy_mc_control,
        "Q Learning": q_learning
    }
    if agent_name in agent_funcs:
        policy, _ = agent_funcs[agent_name](env)
        agent_mode = True
    else:
        policy = None
        agent_mode = False

    stats = {"win": 0, "loss": 0, "draw": 0}
    while True:
        state = env.reset()
        player_first = policy[state] if agent_mode else None
        enemy_first = random.randint(0, 2)

        if not agent_mode:
            draw_message("Round 1 - Choisissez 0/1/2")
            draw_choices()
            player_first = wait_choice()
        result1 = get_result(player_first, enemy_first)
        draw_result(player_first, enemy_first, result1, 1, stats, is_agent=agent_mode)
        wait_for_key()

        new_state = (player_first, enemy_first)
        player_second = policy.get(new_state, random.randint(0, 2)) if agent_mode else None
        counter = WIN_MAP[player_first]
        enemy_probs = [0.15, 0.15, 0.15]
        enemy_probs[counter] = 0.7
        enemy_second = random.choices([0, 1, 2], weights=enemy_probs)[0]

        if not agent_mode:
            draw_message("Round 2 - Choisissez 0/1/2")
            draw_choices()
            player_second = wait_choice()
        result2 = get_result(player_second, enemy_second)
        draw_result(player_second, enemy_second, result2, 2, stats, is_agent=agent_mode)
        wait_for_key()

        for r in [result1, result2]:
            if r == 1:
                stats["win"] += 1
            elif r == -1:
                stats["loss"] += 1
            else:
                stats["draw"] += 1

        draw_message("🎮 Partie terminée - R pour rejouer | Échap pour quitter")
        draw_score(stats)
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        break
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            else:
                continue
            break


def wait_choice():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_0, pygame.K_1, pygame.K_2]:
                    return int(event.unicode)


def get_result(player, enemy):
    if player == enemy:
        return 0
    elif WIN_MAP[player] == enemy:
        return 1
    else:
        return -1


if __name__ == "__main__":
    agent_name = show_agent_menu()
    rps_game(agent_name)
