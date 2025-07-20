import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

GRAPH_DIR = "pm_graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

file_path = "../Reports/td_agents_comparison.xlsx"
df = pd.read_excel(file_path)


# Cellule 3 : GRAPHS1 - Mean Score + Mean Steps par agent et environnement

envs = df['env'].unique()

for env in envs:
    env_df = df[df['env'] == env]

    fig, ax1 = plt.subplots()

    # Score moyen
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

    # Mean Steps sur axe secondaire
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

# Cellule 4 mise à jour : Impact des hyperparamètres via des barplots avec erreur standard

hyperparams = ["gamma", "alpha", "epsilon"]

for env in envs:
    env_df = df[df["env"] == env]
    for param in hyperparams:
        if param in df.columns:
            plt.figure()
            sns.barplot(
                data=env_df,
                x=param,
                y="mean_score",
                hue="agent",
                ci="sd",
                palette="Set2"
            )
            plt.title(f"Score moyen des agents sur l'environnement {env} selon {param}")
            plt.ylabel("Score moyen")
            plt.xlabel(param.capitalize())
            plt.grid(True, axis="y")
            plt.legend(title="Agent")
            plt.tight_layout()

            filename = os.path.join(GRAPH_DIR, f"{env}_vs_{param}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Graphique sauvegardé : {filename}")

# Cellule 5 (optionnelle) : Résumé des performances par agent
summary = df.groupby("agent")[["mean_score", "mean_steps", "time"]].mean().round(2)
summary = summary.sort_values(by="mean_score", ascending=False)
