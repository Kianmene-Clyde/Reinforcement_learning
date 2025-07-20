import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paramètres ===
EXCEL_PATH = "../Reports/dp_agents_comparison.xlsx"
FIGURE_DIR = "../Reports/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# === Lecture des résultats ===
try:
    df = pd.read_excel(EXCEL_PATH)
except FileNotFoundError:
    print(f"Fichier non trouvé : {EXCEL_PATH}")
    exit()

# === Nettoyage ===
df_clean = df.dropna(subset=["mean_score"])

# === Barplot global pour chaque environnement ===
for env in df_clean["env"].unique():
    env_data = df_clean[df_clean["env"] == env]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=env_data,
        x="gamma",
        y="mean_score",
        hue="agent",
        ci="sd"
    )
    plt.title(f"Score moyen des agents sur l'environnement {env}")
    plt.xlabel("Gamma")
    plt.ylabel("Score moyen")
    plt.legend(title="Agent")
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, f"{env}_barplot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Barplot enregistré : {save_path}")

# === Heatmap par agent et environnement ===
for env in df_clean["env"].unique():
    for agent in df_clean["agent"].unique():
        subset = df_clean[(df_clean["env"] == env) & (df_clean["agent"] == agent)]
        if subset.empty:
            continue

        pivot = subset.pivot(index="theta", columns="gamma", values="mean_score")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"Heatmap des scores | {agent} sur {env}")
        plt.xlabel("Gamma")
        plt.ylabel("Theta")
        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, f"{env}_{agent}_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Heatmap enregistrée : {save_path}")

print("\nToutes les visualisations ont été générées.")
