import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les résultats
file_path = "../Reports/global_comparison.xlsx"
sheet_name = "RésuméGlobal"
output_dir = "../Reports/Visualisations"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(file_path, sheet_name=sheet_name)
df.columns = [col.lower() for col in df.columns]

# 1. Performance moyenne par agent/environnement
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="env", y="mean_score", hue="agent")
plt.title("Performance moyenne des agents par environnement")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_performance_agents_envs.png"))
plt.close()

# 2. Boxplots des scores par agent
for agent in df['agent'].unique():
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[df['agent'] == agent], x='env', y='mean_score')
    plt.title(f"Distribution des scores pour {agent}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{agent}.png"))
    plt.close()

# 3. Heatmap alpha vs epsilon (par agent/env)
for agent in df['agent'].unique():
    for env in df['env'].unique():
        subset = df[(df['agent'] == agent) & (df['env'] == env)]
        if 'alpha' in subset.columns and 'epsilon' in subset.columns:
            pivot = subset.pivot_table(index='alpha', columns='epsilon', values='mean_score', aggfunc='mean')
            if pivot.size > 1:
                plt.figure(figsize=(8, 6))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title(f"{agent} sur {env} (alpha vs epsilon)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{agent}_{env}_alpha_epsilon_heatmap.png"))
                plt.close()

# 4. Score moyen vs num_episodes
if 'num_episodes' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='num_episodes', y='mean_score', hue='agent', style='env', markers=True)
    plt.title("Score moyen en fonction du nombre d'épisodes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_vs_num_episodes.png"))
    plt.close()

# 5. Corrélation hyperparamètres vs score
corr_params = ['gamma', 'alpha', 'epsilon', 'theta', 'planning_steps', 'kappa']
df_corr = df[['mean_score'] + [p for p in corr_params if p in df.columns]]
corr = df_corr.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis')
plt.title("Corrélation entre les hyperparamètres et le score moyen")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_hyperparams_score.png"))
plt.close()

print("Graphiques de visualisation générés dans Reports/Visualisations/")
