import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pygame
import sys

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

# === Pygame ===
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Visualisation RL")
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 20)
WHITE, BLACK, GREY, BLUE, GREEN = (255, 255, 255), (0, 0, 0), (220, 220, 220), (100, 100, 255), (50, 200, 50)

step = 0
selected_agent = 0
selected_env = 0
selected_graph = 0
running = True

while running:
    screen.fill(WHITE)
    title = ["Choisissez un agent", "Choisissez un environnement", "Choisissez un graphique"][step]
    screen.blit(font.render(title, True, BLACK), (WIDTH // 2 - 150, 30))

    options = [agents, envs, graph_options][step]
    for i, opt in enumerate(options):
        color = GREEN if i == [selected_agent, selected_env, selected_graph][step] else GREY
        pygame.draw.rect(screen, color, (100, 80 + i * 40, 600, 35), border_radius=5)
        label = small_font.render(f"{i + 1} - {opt}", True, BLACK)
        screen.blit(label, (110, 85 + i * 40))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.unicode.isdigit():
                idx = int(event.unicode) - 1
                if step == 0 and 0 <= idx < len(agents):
                    selected_agent = idx
                    step += 1
                elif step == 1 and 0 <= idx < len(envs):
                    selected_env = idx
                    step += 1
                elif step == 2 and 0 <= idx < len(graph_options):
                    selected_graph = idx
                    running = False

# === Génération du graphique ===
agent = agents[selected_agent]
env = envs[selected_env]
graph = graph_options[selected_graph]
subset = df[(df['agent'] == agent) & (df['env'] == env)]

if selected_graph == 0:  # Boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=subset, x='env', y='mean_score')
    plt.title(f"Boxplot des scores de {agent} sur {env}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{agent}_{env}.png"))

elif selected_graph == 1:  # Heatmap alpha/epsilon
    if 'alpha' in subset.columns and 'epsilon' in subset.columns:
        pivot = subset.pivot_table(index='alpha', columns='epsilon', values='mean_score', aggfunc='mean')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Heatmap alpha/epsilon - {agent} sur {env}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{agent}_{env}.png"))

elif selected_graph == 2:  # Courbe score vs episodes
    if 'num_episodes' in subset.columns:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=subset, x='num_episodes', y='mean_score', marker='o')
        plt.title(f"Score vs num_episodes - {agent} sur {env}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"lineplot_{agent}_{env}.png"))

elif selected_graph == 3:  # Corrélation
    corr_params = ['gamma', 'alpha', 'epsilon', 'theta', 'planning_steps', 'kappa']
    available = [p for p in corr_params if p in subset.columns]
    df_corr = subset[['mean_score'] + available]
    if len(df_corr.columns) > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap='viridis')
        plt.title(f"Corrélation hyperparamètres vs score - {agent} sur {env}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"correlation_{agent}_{env}.png"))

print(f"\n✅ Graphique '{graph}' généré pour {agent} sur {env} dans {OUTPUT_DIR}")
