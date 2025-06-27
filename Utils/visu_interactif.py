import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pygame

# === Paramètres ===
FILE_PATH = "Reports/global_comparison.xlsx"
SHEET_NAME = "Résultats"
OUTPUT_DIR = "Reports/Visualisations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Chargement des données ===
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = [col.lower() for col in df.columns]

agents = sorted(df['agent'].unique())
envs = sorted(df['env'].unique())

graph_options = [
    "Boxplot des scores",
    "Heatmap alpha vs epsilon",
    "Courbe score vs num_episodes",
    "Corrélation entre hyperparamètres et score"
]

# === Pygame setup ===
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Visualisation Interactives des Agents RL")
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 20)
WHITE, BLACK, GREY, GREEN = (255, 255, 255), (0, 0, 0), (230, 230, 230), (50, 200, 50)

step, selected = 0, [0, 0, 0]
running = True


def draw_screen(step, options, title):
    screen.fill(WHITE)
    screen.blit(font.render(title, True, BLACK), (WIDTH // 2 - 150, 30))
    for i, opt in enumerate(options):
        color = GREEN if i == selected[step] else GREY
        pygame.draw.rect(screen, color, (100, 80 + i * 40, 600, 35), border_radius=5)
        label = small_font.render(f"{i + 1} - {opt}", True, BLACK)
        screen.blit(label, (110, 85 + i * 40))
    pygame.display.flip()


while running:
    options = [agents, envs, graph_options][step]
    title = ["Choisissez un agent", "Choisissez un environnement", "Choisissez un graphique"][step]
    draw_screen(step, options, title)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit();
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.unicode.isdigit():
                idx = int(event.unicode) - 1
                if 0 <= idx < len(options):
                    selected[step] = idx
                    if step < 2:
                        step += 1
                    else:
                        running = False
            elif event.key == pygame.K_DOWN:
                selected[step] = (selected[step] + 1) % len(options)
            elif event.key == pygame.K_UP:
                selected[step] = (selected[step] - 1) % len(options)
            elif event.key == pygame.K_RETURN:
                if step < 2:
                    step += 1
                else:
                    running = False

# === Génération du graphique ===
agent = agents[selected[0]]
env = envs[selected[1]]
graph = graph_options[selected[2]]
subset = df[(df['agent'] == agent) & (df['env'] == env)]

try:
    if selected[2] == 0:  # Boxplot
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=subset, x='env', y='mean_score')
        plt.title(f"Boxplot des scores de {agent} sur {env}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{agent}_{env}.png"))

    elif selected[2] == 1:  # Heatmap
        if 'alpha' in subset.columns and 'epsilon' in subset.columns:
            pivot = subset.pivot_table(index='alpha', columns='epsilon', values='mean_score', aggfunc='mean')
            if pivot.size > 1:
                plt.figure(figsize=(8, 6))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title(f"Heatmap alpha/epsilon - {agent} sur {env}")
                plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{agent}_{env}.png"))

    elif selected[2] == 2:  # Courbe score vs num_episodes
        if 'num_episodes' in subset.columns:
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=subset, x='num_episodes', y='mean_score', marker='o')
            plt.title(f"Score vs num_episodes - {agent} sur {env}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"lineplot_{agent}_{env}.png"))

    elif selected[2] == 3:  # Corrélation hyperparamètres vs score
        corr_params = ['gamma', 'alpha', 'epsilon', 'theta', 'planning_steps', 'kappa']
        available = [p for p in corr_params if p in subset.columns]
        if available:
            df_corr = subset[['mean_score'] + available]
            plt.figure(figsize=(10, 6))
            sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap='viridis')
            plt.title(f"Corrélation hyperparamètres vs score - {agent} sur {env}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"correlation_{agent}_{env}.png"))

    print(f"\n✅ Graphique '{graph}' généré pour {agent} sur {env} dans {OUTPUT_DIR}")

except Exception as e:
    print(f"\n❌ Une erreur est survenue lors de la génération du graphique : {e}")
