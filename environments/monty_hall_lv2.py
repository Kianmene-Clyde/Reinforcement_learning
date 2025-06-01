import random
import pygame
import sys
import time


# === ENVIRONNEMENT MONTY HALL LV2 ===
class MontyHallEnvLv2:
    def __init__(self):
        self.doors = list(range(5))
        self.reset()

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.chosen_door = None
        self.revealed_doors = []
        self.state = "start"
        return self.state

    def step(self, action):
        if self.state == "start":
            self.chosen_door = action
            self.revealed_doors = random.sample(
                [d for d in self.doors if d != self.chosen_door and d != self.winning_door], 3
            )
            self.state = "reveal"
            return ("reveal", self.chosen_door, self.revealed_doors), 0.0, False
        elif self.state == "reveal":
            final_choice = action
            reward = 1.0 if final_choice == self.winning_door else 0.0
            self.state = "done"
            return ("done", final_choice), reward, True


# === AFFICHAGE PYGAME ===
WIDTH, HEIGHT = 800, 400
WHITE, BLACK, GREEN, RED, GREY = (255, 255, 255), (0, 0, 0), (50, 200, 50), (255, 80, 80), (200, 200, 200)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 22)
pygame.display.set_caption("Monty Hall - Niveau 2 (5 Portes)")


def draw_start(win, loss):
    screen.fill(WHITE)
    screen.blit(font.render("Choisissez une porte (0 à 4)", True, BLACK), (260, 20))
    for i in range(5):
        pygame.draw.rect(screen, GREY, (60 + i * 140, 100, 100, 150))
        screen.blit(font.render(str(i), True, BLACK), (100 + i * 140, 160))
    draw_score(win, loss)
    pygame.display.flip()


def draw_reveal(chosen, revealed, win, loss):
    screen.fill(WHITE)
    screen.blit(font.render("3 portes éliminées. Choisissez votre porte finale", True, BLACK), (160, 20))
    for i in range(5):
        color = RED if i in revealed else GREEN if i == chosen else GREY
        pygame.draw.rect(screen, color, (60 + i * 140, 100, 100, 150))
        screen.blit(font.render(str(i), True, BLACK), (100 + i * 140, 160))
    draw_score(win, loss)
    pygame.display.flip()


def draw_result(winning, final, reward, win, loss):
    screen.fill(WHITE)
    msg = "✅ GAGNÉ !" if reward == 1.0 else "❌ PERDU"
    screen.blit(font.render(f"{msg} Gagnante: {winning} | Votre choix: {final}  (R pour rejouer)", True, BLACK),
                (100, 20))
    for i in range(5):
        color = GREEN if i == winning else GREY
        pygame.draw.rect(screen, color, (60 + i * 140, 100, 100, 150))
        screen.blit(font.render(str(i), True, BLACK), (100 + i * 140, 160))
    draw_score(win, loss)
    pygame.display.flip()


def draw_score(win, loss):
    pygame.draw.rect(screen, (245, 245, 245), (0, HEIGHT - 40, WIDTH, 40))
    screen.blit(font.render(f"✅ Victoires : {win}", True, (0, 150, 0)), (20, HEIGHT - 30))
    screen.blit(font.render(f"❌ Défaites : {loss}", True, (200, 0, 0)), (WIDTH // 2, HEIGHT - 30))


# === LOOP PRINCIPALE ===
def main(agent_mode=False):
    env = MontyHallEnvLv2()
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
            draw_reveal(env.chosen_door, env.revealed_doors, win_count, loss_count)
        elif env.state == "done":
            if not scored:
                final_choice = step_result[0][1]
                reward = step_result[1]
                win_count += int(reward == 1.0)
                loss_count += int(reward == 0.0)
                scored = True
            draw_result(env.winning_door, final_choice, reward, win_count, loss_count)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif not agent_mode and event.type == pygame.KEYDOWN:
                if env.state == "start" and event.unicode in ["0", "1", "2", "3", "4"]:
                    step_result = env.step(int(event.unicode))
                elif env.state == "reveal" and event.unicode in ["0", "1", "2", "3", "4"]:
                    a = int(event.unicode)
                    if a not in env.revealed_doors:
                        step_result = env.step(a)
                elif env.state == "done" and event.key == pygame.K_r:
                    env.reset()
                    step_result = None
                    scored = False

        if agent_mode:
            time.sleep(1)
            if env.state == "start":
                step_result = env.step(random.choice(env.doors))
            elif env.state == "reveal":
                remaining = [d for d in env.doors if d not in env.revealed_doors]
                step_result = env.step(random.choice(remaining))
            elif env.state == "done":
                time.sleep(1)
                env.reset()
                step_result = None
                scored = False

    pygame.quit()
    sys.exit()


# === MENU ===
print("1 - Agent automatique")
print("2 - Joueur humain")
choice = input("Choix : ").strip()
main(agent_mode=(choice == "1"))
