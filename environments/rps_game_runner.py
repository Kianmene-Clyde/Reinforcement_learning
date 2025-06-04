import pygame
import sys
import random

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, BLUE = (0, 180, 0), (180, 0, 0), (0, 0, 180)
GREY = (230, 230, 230)
WIDTH, HEIGHT = 700, 400
CHOICES = ["Pierre", "Feuille", "Ciseaux"]
WIN_MAP = {0: 2, 1: 0, 2: 1}


class RPSGameRunner:
    def __init__(self, agent_name="Policy Iteration"):
        pygame.init()
        self.agent_name = agent_name
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Rock Paper Scissors - Score Cumulé")
        self.font = pygame.font.SysFont("Arial", 24)

    def draw_message(self, text):
        self.screen.fill(WHITE)
        label = self.font.render(text, True, BLACK)
        self.screen.blit(label, (WIDTH // 2 - label.get_width() // 2, 40))
        pygame.display.flip()

    def draw_choices(self):
        self.screen.fill(WHITE)
        for i in range(3):
            pygame.draw.rect(self.screen, GREY, (100 + i * 180, 150, 120, 100))
            pygame.draw.rect(self.screen, BLACK, (100 + i * 180, 150, 120, 100), 2)
            label = self.font.render(f"{i} = {CHOICES[i]}", True, BLACK)
            self.screen.blit(label, (110 + i * 180, 190))
        pygame.display.flip()

    def draw_result(self, player, enemy, result, round_num, stats, is_agent=False):
        self.screen.fill(WHITE)
        result_text = "ÉGALITÉ" if result == 0 else "✅ GAGNÉ !" if result == 1 else "❌ PERDU"
        who = "Agent" if is_agent else "Vous"
        text = f"Round {round_num} - {who}: {CHOICES[player]} vs Ennemi: {CHOICES[enemy]} ➜ {result_text}"
        color = GREEN if result == 1 else RED if result == -1 else BLACK
        label = self.font.render(text, True, color)
        self.screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - 30))
        self.draw_score(stats)
        pygame.display.flip()

    def draw_score(self, stats):
        win, loss, draw = stats["win"], stats["loss"], stats["draw"]
        self.screen.blit(self.font.render(f"✅ Victoires : {win}", True, GREEN), (20, HEIGHT - 60))
        self.screen.blit(self.font.render(f"❌ Défaites : {loss}", True, RED), (250, HEIGHT - 60))
        self.screen.blit(self.font.render(f"🤝 Égalités : {draw}", True, BLUE), (480, HEIGHT - 60))

    def wait_for_key(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    return
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def run(self):
        print("1 - Agent automatique")
        print("2 - Joueur humain")
        choice = input("Choix : ").strip()
        if choice == "1":
            self.play(agent_mode=True)
        elif choice == "2":
            self.play(agent_mode=False)
        else:
            print("Choix invalide.")

    def play(self, agent_mode=False):
        stats = {"win": 0, "loss": 0, "draw": 0}
        running = True

        while running:
            scores = []
            first_choice = None
            enemy_first = None
            round_num = 1

            # Round 1
            self.draw_message(f"Round {round_num} - Choisissez : 0 = Pierre, 1 = Feuille, 2 = Ciseaux")
            self.draw_choices()

            if agent_mode:
                pygame.time.delay(1000)
                first_choice = random.choice([0, 1, 2])
                enemy_first = random.randint(0, 2)
            else:
                first_choice = self.wait_choice()
                enemy_first = random.randint(0, 2)

            result1 = self.get_result(first_choice, enemy_first)
            self.draw_result(first_choice, enemy_first, result1, round_num, stats, is_agent=agent_mode)
            scores.append(result1)
            self.wait_for_key()

            # Round 2
            round_num += 1
            self.draw_message(f"Round {round_num} - Choisissez : 0 = Pierre, 1 = Feuille, 2 = Ciseaux")
            self.draw_choices()

            if agent_mode:
                pygame.time.delay(1000)
                second_choice = random.choice([0, 1, 2])
                counter = WIN_MAP[first_choice]
                enemy_probs = [0.15, 0.15, 0.15]
                enemy_probs[counter] = 0.7
                enemy_second = random.choices([0, 1, 2], weights=enemy_probs)[0]
            else:
                second_choice = self.wait_choice()
                counter = WIN_MAP[first_choice]
                enemy_probs = [0.15, 0.15, 0.15]
                enemy_probs[counter] = 0.7
                enemy_second = random.choices([0, 1, 2], weights=enemy_probs)[0]

            result2 = self.get_result(second_choice, enemy_second)
            self.draw_result(second_choice, enemy_second, result2, round_num, stats, is_agent=agent_mode)
            scores.append(result2)
            self.wait_for_key()

            # Score update
            for res in scores:
                if res == 1:
                    stats["win"] += 1
                elif res == -1:
                    stats["loss"] += 1
                else:
                    stats["draw"] += 1

            self.draw_message("🎮 Partie terminée - R pour rejouer | Échap pour quitter")
            self.draw_score(stats)

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

    def wait_choice(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.unicode in "012":
                    return int(event.unicode)

    def get_result(self, player, enemy):
        if player == enemy:
            return 0
        elif WIN_MAP[player] == enemy:
            return 1
        else:
            return -1
