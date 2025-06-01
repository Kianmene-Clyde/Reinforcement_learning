import random
import pygame
import sys
import time


# === ENVIRONNEMENT MONTY HALL ===
class MontyHallEnv:
    def __init__(self):
        self.doors = [0, 1, 2]
        self.reset()

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.state = "start"
        self.chosen_door = None
        self.revealed_door = None
        return self.state

    def get_actions(self):
        if self.state == "start":
            return self.doors
        elif self.state == "reveal":
            return [0, 1]  # 0 = garder, 1 = changer
        return []

    def step(self, action):
        if self.state == "start":
            self.chosen_door = action
            self.revealed_door = random.choice(
                [d for d in self.doors if d != self.chosen_door and d != self.winning_door]
            )
            self.state = "reveal"
            return ("reveal", self.chosen_door, self.revealed_door), 0.0, False

        elif self.state == "reveal":
            final_choice = self.chosen_door if action == 0 else (
                [d for d in self.doors if d not in [self.chosen_door, self.revealed_door]][0]
            )
            reward = 1.0 if final_choice == self.winning_door else 0.0
            done = True
            return ("done", final_choice), reward, done


# === AFFICHAGE PYGAME ===
WHITE, BLACK, BLUE, GREEN, RED, GREY = (255, 255, 255), (0, 0, 0), (50, 50, 255), (50, 200, 50), (255, 80, 80), (200,
                                                                                                                 200,
                                                                                                                 200)
WIDTH, HEIGHT = 600, 300

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 24)
pygame.display.set_caption("Monty Hall - Niveau 1")


def draw_start(win_count, loss_count):
    screen.fill(WHITE)
    label = font.render("Choisissez une porte (0, 1, 2)", True, BLACK)
    screen.blit(label, (150, 20))
    for i in range(3):
        pygame.draw.rect(screen, GREY, (100 + i * 150, 100, 100, 150))
        num = font.render(str(i), True, BLACK)
        screen.blit(num, (140 + i * 150, 160))
    draw_score(win_count, loss_count)
    pygame.display.flip()


def draw_reveal(chosen, revealed, win_count, loss_count):
    screen.fill(WHITE)
    label = font.render(f"Porte {revealed} éliminée. Choisissez : 0 = garder, 1 = changer", True, BLACK)
    screen.blit(label, (40, 20))
    for i in range(3):
        color = RED if i == revealed else GREEN if i == chosen else GREY
        pygame.draw.rect(screen, color, (100 + i * 150, 100, 100, 150))
        num = font.render(str(i), True, BLACK)
        screen.blit(num, (140 + i * 150, 160))
    draw_score(win_count, loss_count)
    pygame.display.flip()


def draw_result(winning_door, final_choice, reward, win_count, loss_count):
    screen.fill(WHITE)
    text = "✅ GAGNÉ !" if reward == 1.0 else "❌ PERDU"
    label = font.render(f"{text} - Gagnante: {winning_door}, Votre choix: {final_choice} | R = Rejouer", True, BLACK)
    screen.blit(label, (30, 20))
    for i in range(3):
        color = GREEN if i == winning_door else GREY
        pygame.draw.rect(screen, color, (100 + i * 150, 100, 100, 150))
        num = font.render(str(i), True, BLACK)
        screen.blit(num, (140 + i * 150, 160))
    draw_score(win_count, loss_count)
    pygame.display.flip()


def draw_score(win_count, loss_count):
    pygame.draw.rect(screen, (245, 245, 245), (0, HEIGHT - 40, WIDTH, 40))
    win_label = font.render(f"✅ Victoires : {win_count}", True, (0, 150, 0))
    loss_label = font.render(f"❌ Défaites : {loss_count}", True, (200, 0, 0))
    screen.blit(win_label, (20, HEIGHT - 30))
    screen.blit(loss_label, (320, HEIGHT - 30))


# === LOOP PRINCIPALE ===
def main(agent_mode=False):
    env = MontyHallEnv()
    state = env.reset()
    step_result = None
    win_count = 0
    loss_count = 0
    scored = False
    running = True

    while running:
        if env.state == "start":
            draw_start(win_count, loss_count)
        elif env.state == "reveal":
            draw_reveal(env.chosen_door, env.revealed_door, win_count, loss_count)
        elif env.state == "done":
            if not scored:
                final_choice = step_result[0][1]
                reward = step_result[1]
                if reward == 1.0:
                    win_count += 1
                else:
                    loss_count += 1
                scored = True
            draw_result(env.winning_door, final_choice, reward, win_count, loss_count)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif not agent_mode and event.type == pygame.KEYDOWN:
                if env.state == "start" and event.unicode in ["0", "1", "2"]:
                    a = int(event.unicode)
                    step_result = env.step(a)
                elif env.state == "reveal" and event.unicode in ["0", "1"]:
                    a = int(event.unicode)
                    step_result = env.step(a)
                    env.state = "done"
                elif env.state == "done" and event.key == pygame.K_r:
                    state = env.reset()  # ✅ AJOUT ICI
                    step_result = None
                    scored = False

        if agent_mode:
            pygame.time.delay(1500)
            if env.state == "start":
                step_result = env.step(random.choice([0, 1, 2]))
            elif env.state == "reveal":
                step_result = env.step(1)  # l’agent change toujours
                env.state = "done"
            elif env.state == "done":
                state = env.reset()  # ✅ AJOUT ICI AUSSI
                step_result = None
                scored = False

    pygame.quit()
    sys.exit()


# === MENU ===
print("1 - Agent automatique")
print("2 - Joueur humain (0/1/2 puis 0=garder ou 1=changer)")
choice = input("Choix : ").strip()
main(agent_mode=(choice == "1"))
