import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Création du dossier pour les graphes
GRAPH_DIR = "pm_graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

file_path = "../Reports/planning_agents_comparison.xlsx"
df = pd.read_excel(file_path)

envs = df['env'].dropna().unique()

# GRAPHE 1 : Mean Score + Mean Steps pour chaque agent dans chaque environnement
for env in envs:
    env_df = df[df['env'] == env]

    fig, ax1 = plt.subplots()

    # Barplot du score moyen
    sns.barplot(
        data=env_df,
        x="agent",
        y="mean_score",
        ci=None,
        ax=ax1,
        palette="Blues_d"
    )
    ax1.set_ylabel("Mean Score", color="blue")
    ax1.set_title(f"Performance par Agent dans {env}")
    ax1.set_xlabel("Agent")

    # Courbe du mean_steps sur axe secondaire
    ax2 = ax1.twinx()
    sns.pointplot(
        data=env_df,
        x="agent",
        y="mean_steps",
        color="red",
        ax=ax2
    )
    ax2.set_ylabel("Mean Steps", color="red")

    plt.tight_layout()
    filename = os.path.join(GRAPH_DIR, f"performance_{env}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Graphique sauvegardé : {filename}")

# GRAPHE 2 : Impact des hyperparamètres sur le score moyen
hyperparams = ["gamma", "alpha", "epsilon", "planning_steps", "kappa"]

for env in envs:
    env_df = df[df["env"] == env]
    for param in hyperparams:
        if param in df.columns and not df[param].isnull().all():
            plt.figure()
            sns.barplot(
                data=env_df,
                x=param,
                y="mean_score",
                hue="agent",
                ci="sd",
                palette="Set2"
            )
            plt.title(f"Impact de {param} sur Score moyen ({env})")
            plt.ylabel("Score moyen")
            plt.xlabel(param.capitalize())
            plt.grid(True, axis="y")
            plt.legend(title="Agent")
            plt.tight_layout()

            filename = os.path.join(GRAPH_DIR, f"{env}_vs_{param}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Graphique sauvegardé : {filename}")

summary = df.groupby("agent")[["mean_score", "mean_steps", "time"]].mean().round(2)
print("\nRésumé des performances globales :")
print(summary.sort_values(by="mean_score", ascending=False))
