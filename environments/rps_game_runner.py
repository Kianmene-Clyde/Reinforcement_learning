import pygame
import sys
import random
from datetime import datetime
import pandas as pd
import os

from Utils.save_load_policy import save_policy, load_policy
from Utils.export_results_to_xlsx import export_results

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


def safe_render(text, color):
    try:
        return font.render(text, True, color)
    except UnicodeEncodeError:
        clean_text = text.encode("ascii", "ignore").decode()
        return font.render(clean_text, True, color)


def draw_message(text):
    screen.fill(WHITE)
    label = safe_render(text, BLACK)
    screen.blit(label, (WIDTH // 2 - label.get_width() // 2, 40))
    pygame.display.flip()


def draw_choices():
    screen.fill(WHITE)
    for i in range(3):
        pygame.draw.rect(screen, GREY, (100 + i * 180, 150, 120, 100))
        pygame.draw.rect(screen, BLACK, (100 + i * 180, 150, 120, 100), 2)
        label = safe_render(f"{i} = {CHOICES[i]}", BLACK)
        screen.blit(label, (110 + i * 180, 190))
    pygame.display.flip()


def draw_result(player, enemy, result, round_num, stats, is_agent=False):
    screen.fill(WHITE)
    result_text = "ÉGALITÉ" if result == 0 else "✔ GAGNÉ !" if result == 1 else "✘ PERDU"
    who = "Agent" if is_agent else "Vous"
    text = f"Round {round_num} - {who}: {CHOICES[player]} vs Ennemi: {CHOICES[enemy]} -> {result_text}"
    color = GREEN if result == 1 else RED if result == -1 else BLACK
    label = safe_render(text, color)
    screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - 30))
    draw_score(stats)
    pygame.display.flip()


def draw_score(stats):
    win, loss, draw = stats["win"], stats["loss"], stats["draw"]
    screen.blit(safe_render(f"✔ Victoires : {win}", GREEN), (20, HEIGHT - 60))
    screen.blit(safe_render(f"✘ Défaites : {loss}", RED), (250, HEIGHT - 60))
    screen.blit(safe_render(f"= Égalités : {draw}", BLUE), (480, HEIGHT - 60))


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
        key_label = f"{i + 1}" if i < 9 else "0" if i == 9 else "A"
        pygame.draw.rect(screen, GREY, (150, 70 + i * 30, 400, 30))
        label = safe_render(f"{key_label} - {text}", BLACK)
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


def print_policy(policy):
    print("\n--- Politique Apprise ---")
    for state, action in policy.items():
        print(f"État {state} -> Action {action} ({CHOICES[action]})")


def rps_game(agent_name="Joueur Humain"):
    env = RPSGameEnv()
    agent_funcs = {
        "Policy Iteration": (policy_iteration, {}),
        "Value Iteration": (value_iteration, {}),
        "Dyna Q": (dyna_q, {"planning_steps": 10}),
        "Dyna Q+": (dyna_q_plus, {"planning_steps": 10, "kappa": 0.01}),
        "Sarsa": (sarsa, {"alpha": 0.1, "epsilon": 0.1, "episodes": 1000}),
        "Expected Sarsa": (expected_sarsa, {"alpha": 0.1, "epsilon": 0.1, "episodes": 1000}),
        "First Visit MC": (on_policy_first_visit_mc_control, {"epsilon": 0.1, "episodes": 1000}),
        "MC ES": (monte_carlo_es, {"episodes": 1000}),
        "Off-policy MC": (off_policy_mc_control, {"episodes": 1000}),
        "Q Learning": (q_learning, {"alpha": 0.1, "epsilon": 0.1, "episodes": 1000})
    }

    filename = f"rps_{agent_name.replace(' ', '_').lower()}.pkl"

    if agent_name in agent_funcs:
        print(f"Agent : {agent_name}")
        print("Souhaitez-vous :")
        print("1 - Charger une politique existante")
        print("2 - Entraîner une nouvelle politique")
        choix = input("Votre choix (1/2) : ").strip()

        if choix == "1":
            try:
                policy = load_policy(filename)
                params = {}
                print_policy(policy)
            except FileNotFoundError as e:
                print(e)
                return
        else:
            agent_func, params = agent_funcs[agent_name]
            policy, _ = agent_func(env, **params)
            print_policy(policy)
            print("Souhaitez-vous sauvegarder cette politique ? (O/N)")
            if input().strip().lower() == "o":
                save_policy(policy, filename)
        agent_mode = True
    else:
        policy = None
        params = {}
        agent_mode = False

    stats = {"win": 0, "loss": 0, "draw": 0}
    while True:
        state = env.reset()
        state_index = env.get_state()
        player_first = policy[state_index] if agent_mode else None
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

        export_results(agent_name, stats, params, params)

        draw_message("Partie terminée - R pour rejouer | Échap pour quitter")
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


if __name__ == "__main__":
    agent_name = show_agent_menu()
    rps_game(agent_name)
