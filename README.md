# Projet Apprentissage par Renforcement (Reinforcement Learning)

## Objectif pédagogique

Ce projet vise à :

- Implémenter et comparer des algorithmes fondamentaux d'apprentissage par renforcement (RL).
- Appliquer ces agents à plusieurs environnements conçus sur mesure (LineWorld, GridWorld, Monty Hall, etc.) et sur des
  environnement secrets.
- Étudier l'impact des hyperparamètres sur les performances des politiques apprises.
- Identifier les meilleures stratégies pour chaque environnement.
- Savoir quand et pourquoi utiliser chaque méthode (DP, Monte Carlo, TD, Planning).

---

## Structure du projet

```
RL_Project/
│
├── agents/                   # Tous les algorithmes RL implémentés
│   ├── dynamic_programming.py
│   ├── monte_carlo_methods.py
│   ├── temporal_difference_methods.py
│   └── planning_methods.py
│
├── environments/             # Tous les environnements RL (y compris secrets)
│   ├── line_world_env.py
│   ├── grid_world_env.py
│   ├── monty_hall_lv1_env.py
│   ├── monty_hall_lv2_env.py
│   ├── rps_game_env.py
│   └── secret_env_*.py       # (fournis plus tard)
│
├── utils/                    # Fonctions auxiliaires
│   ├── export_results.py     # Export Excel
│   ├── save_load_policy.py   # Sauvegarde/chargement de politiques
│   └── visualize_results.py  # Génération de graphes
│
├── main.py                   # Menu principal Pygame (agent vs humain)
├── experiments.py            # Teste tous les agents sur tous les environnements
├── Reports/                  # Exports .xlsx + visualisations
└── README.md                 # Ce fichier
```

---

## Agents implémentés

| Catégorie           | Méthodes                                                                      |
|---------------------|-------------------------------------------------------------------------------|
| Dynamic Programming | `policy_iteration`, `value_iteration`                                         |
| Monte Carlo         | `on_policy_first_visit_mc_control`, `monte_carlo_es`, `off_policy_mc_control` |
| Temporal Difference | `sarsa`, `q_learning`, `expected_sarsa`                                       |
| Planning            | `dyna_q`, `dyna_q_plus`                                                       |

---

## Environnements simulés

| Nom             | Description courte                                                  |
|-----------------|---------------------------------------------------------------------|
| LineWorld       | Environnement linéaire terminal                                     |
| GridWorld       | Grille 2D avec états terminaux                                      |
| Monty Hall LV1  | Problème classique des 3 portes, choix initial puis switch          |
| Monty Hall LV2  | Variante à 5 portes avec 4 étapes                                   |
| RPS Game        | Jeu Pierre-Feuille-Ciseaux sur 2 tours, avec adversaire stratégique |
| SecretEnv (0–3) | Environnements mystères fournis par l'enseignant en fin de projet   |

Tous les environnements user made incluent :

- une interface Pygame
- un mode manuel (humain) et automatique (agent entraîné)
- une visualisation pas-à-pas
- un affichage des flèches de politique ou résultats interactifs

---

## Méthodologie expérimentale

- Tous les agents sont testés sur tous les environnements**
- Une grille d'hyperparamètres est explorée automatiquement pour chaque famille d'agents :
- Les scores moyens et paramètres optimaux sont exportés dans `Reports/global_comparison.xlsx` pour les environnemnt
  user made et dand `SecretReports/secret_comparison.xlsx` pour les environnements secrets.
- Un script de visualisation permet de générer tous les graphes pertinents (boxplots, heatmaps, courbes,
  corrélations)
- Les politiques apprises sont sauvegardables et réutilisables

---

## Sauvegarde / Chargement de politiques

```python
from Utils.save_load_policy import save_policy, load_policy

# Sauvegarder
save_policy(policy, "policy_gridworld_qlearning.pkl")

# Charger
policy = load_policy("policy_gridworld_qlearning.pkl")
```

➡Permet de rejouer une stratégie sans réentraînement.

---

## Visualisation des résultats

Lancer :

```bash
python utils/visualize_results.py
```

Permet de générer :

-  Boxplots par agent et environnement
-  Heatmaps hyperparamètres (alpha vs epsilon, etc.)
-  Courbes de convergence (score vs épisodes)
-  Corrélation entre hyperparamètres et performance

Toutes les images sont stockées dans `Reports/Visualisations/`.

---

## Interprétation des résultats

| Agent              | Environnements préférés        | Explication                                         |
|--------------------|--------------------------------|-----------------------------------------------------|
| `policy_iteration` | `line_world`, `grid_world`     | converge vite avec modèle parfait                   |
| `q_learning`       | `monty_hall_lv1`, `rps_game`   | efficace avec exploration contrôlée                 |
| `dyna_q_plus`      | `monty_hall_lv2`, `grid_world` | avantage avec planification et exploration différée |


# Résumé des Résultats des Expériences RL

## Objectif

Comparer les performances de différents agents RL sur plusieurs environnements avec variation d'hyperparamètres pour
identifier les meilleures combinaisons.

---

## Meilleurs Couples Agent / Environnement

| Environnement  | Agent Optimal    | Score Moyen | Hyperparamètres  |
|----------------|------------------|-------------|------------------|
| LineWorld      | Policy Iteration | XX.XX       | gamma=0.99, ...  |
| GridWorld      | Value Iteration  | XX.XX       | gamma=0.9, ...   |
| Monty Hall LV1 | Q-learning       | XX.XX       | epsilon=0.2, ... |
| Monty Hall LV2 | Dyna-Q+          | XX.XX       | planning=10, ... |
| RPS Game       | Expected SARSA   | XX.XX       | alpha=0.5, ...   |

---

## Analyse Visuelle Automatique

Lance `visualize_results.py` pour :

- Générer des **boxplots** par agent/environnement
- Générer des **heatmaps** des hyperparamètres (alpha vs epsilon)
- Tracer des **courbes score vs épisodes**
- Calculer la **corrélation** entre hyperparamètres et performances

---


## Export

- Résultats complets : `Reports/global_comparison.xlsx`
- Graphiques : `Reports/Visuals/*.png`
- Politiques entraînées : `Reports/Policies/*.pkl`

