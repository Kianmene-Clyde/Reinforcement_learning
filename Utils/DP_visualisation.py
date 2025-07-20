import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chargement du fichier Excel
file_path = "../Reports/dp_agents_comparison.xlsx"  # adapte si besoin
df = pd.read_excel(file_path)

# Configuration d’affichage
sns.set(style="whitegrid")
output_dir = "./dp_graphs"
os.makedirs(output_dir, exist_ok=True)

# Boucle pour chaque environnement
environments = df["env"].unique()
for env in environments:
    plt.figure(figsize=(10, 6))
    subset = df[df["env"] == env]

    # Regrouper pour obtenir les moyennes
    pivot = subset.groupby("agent")[["mean_score", "mean_steps"]].mean().reset_index()
    pivot_melted = pivot.melt(id_vars="agent", value_vars=["mean_score", "mean_steps"],
                              var_name="Métrique", value_name="Valeur")

    sns.barplot(data=pivot_melted, x="agent", y="Valeur", hue="Métrique", palette="Set2")
    plt.title(f"Comparaison des performances dans {env}")
    plt.ylabel("Valeur moyenne")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{env}_score_steps_comparison.png"))
    plt.show()
