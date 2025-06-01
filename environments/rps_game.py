import pygame
import sys
import random

# === Paramètres ===
WIDTH, HEIGHT = 700, 400
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, BLUE = (0, 180, 0), (180, 0, 0), (0, 0, 180)
GREY = (230, 230, 230)
CHOICES = ["Pierre", "Feuille", "Ciseaux"]
WIN_MAP = {0: 2, 1: 0, 2: 1}

# === Initialisation Pygame ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Paper Scissors - Score Cumulé")
font = pygame.font.SysFont("Arial", 24)


# === Fonctions d'affichage ===
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


# === Jeu Principal ===
def rps_game(agent_mode=False):
    stats = {"win": 0, "loss": 0, "draw": 0}
    running = True

    while running:
        state = "start"
        scores = []
        first_choice = None
        enemy_first = None

        while state != "done":
            if state == "start":
                draw_message("Round 1 - Choisissez : 0 = Pierre, 1 = Feuille, 2 = Ciseaux")
                draw_choices()
                state = "wait_first"

            elif state == "wait_second":
                draw_message("Round 2 - Rejouez : 0 = Pierre, 1 = Feuille, 2 = Ciseaux")
                draw_choices()
                state = "wait_second_choice"

            if agent_mode:
                pygame.time.delay(1000)
                if state == "wait_first":
                    first_choice = random.choice([0, 1, 2])
                    enemy_first = random.randint(0, 2)
                    result1 = 0 if first_choice == enemy_first else 1 if WIN_MAP[first_choice] == enemy_first else -1
                    draw_result(first_choice, enemy_first, result1, 1, stats, is_agent=True)
                    scores.append(result1)
                    pygame.time.delay(2000)
                    state = "wait_second"

                elif state == "wait_second_choice":
                    second_choice = random.choice([0, 1, 2])
                    counter = WIN_MAP[first_choice]
                    enemy_probs = [0.15, 0.15, 0.15]
                    enemy_probs[counter] = 0.7
                    enemy_second = random.choices([0, 1, 2], weights=enemy_probs)[0]
                    result2 = 0 if second_choice == enemy_second else 1 if WIN_MAP[
                                                                               second_choice] == enemy_second else -1
                    draw_result(second_choice, enemy_second, result2, 2, stats, is_agent=True)
                    scores.append(result2)
                    pygame.time.delay(2500)
                    state = "done"

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif not agent_mode and event.type == pygame.KEYDOWN:
                    try:
                        if state == "wait_first" and event.unicode in "012":
                            first_choice = int(event.unicode)
                            enemy_first = random.randint(0, 2)
                            result1 = 0 if first_choice == enemy_first else 1 if WIN_MAP[
                                                                                     first_choice] == enemy_first else -1
                            draw_result(first_choice, enemy_first, result1, 1, stats)
                            scores.append(result1)
                            wait_for_key()
                            state = "wait_second"

                        elif state == "wait_second_choice" and event.unicode in "012":
                            second_choice = int(event.unicode)
                            counter = WIN_MAP[first_choice]
                            enemy_probs = [0.15, 0.15, 0.15]
                            enemy_probs[counter] = 0.7
                            enemy_second = random.choices([0, 1, 2], weights=enemy_probs)[0]
                            result2 = 0 if second_choice == enemy_second else 1 if WIN_MAP[
                                                                                       second_choice] == enemy_second else -1
                            draw_result(second_choice, enemy_second, result2, 2, stats)
                            scores.append(result2)
                            wait_for_key()
                            state = "done"
                    except ValueError:
                        pass

        for res in scores:
            if res == 1:
                stats["win"] += 1
            elif res == -1:
                stats["loss"] += 1
            else:
                stats["draw"] += 1

        draw_message("🎮 Partie terminée - R pour rejouer | Échap pour quitter")
        draw_score(stats)

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()


# === Menu ===
print("1 - Agent automatique")
print("2 - Joueur humain")
choice = input("Choix : ").strip()
rps_game(agent_mode=(choice == "1"))
