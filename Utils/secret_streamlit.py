import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(layout="wide")
st.title("Secret RL Experiments Dashboard")

# === Chargement des données ===
DATA_PATH = "SecretReports/secret_comparison_full.xlsx"


@st.cache_data
def load_data():
    xls = pd.ExcelFile(DATA_PATH)
    sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names if sheet not in ["RésuméGlobal", "BestParams"]}
    global_df = xls.parse("RésuméGlobal")
    best_df = xls.parse("BestParams")
    return sheets, global_df, best_df


sheets, df_all, df_best = load_data()

# === SIDEBAR : Paramètres ===
st.sidebar.title("Paramètres")
selected_env = st.sidebar.selectbox("Choisir un environnement secret", list(sheets.keys()))
df_env = sheets[selected_env]

agents = df_env["agent"].unique().tolist()
selected_agents = st.sidebar.multiselect("🤖 Sélectionner les agents", agents, default=agents)

filtered_df = df_env[df_env["agent"].isin(selected_agents)]

st.sidebar.markdown("---")
st.sidebar.markdown("Filtrage des hyperparamètres")

hyperparams = ["gamma", "alpha", "epsilon", "theta", "planning_steps", "kappa", "episodes"]
for param in hyperparams:
    if param in filtered_df.columns:
        values = sorted(filtered_df[param].unique())
        if len(values) > 1:
            selected_values = st.sidebar.multiselect(f"{param}", values, default=values)
            filtered_df = filtered_df[filtered_df[param].isin(selected_values)]

# === MAIN : Résultats ===
st.markdown(f"### Résultats pour l'environnement secret `{selected_env}`")

# === Barplot des scores ===
st.subheader("Score moyen par agent")
plt.figure(figsize=(10, 5))
sns.barplot(data=filtered_df, x="agent", y="mean_score", ci="sd", palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("Score moyen")
plt.xlabel("Agent")
st.pyplot(plt.gcf())
plt.clf()

# === Barplot des steps ===
if 'mean_steps' in filtered_df.columns:
    st.subheader("Steps moyens par épisode")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=filtered_df, x="agent", y="mean_steps", palette="mako")
    plt.xticks(rotation=45)
    plt.ylabel("Nombre moyen de steps")
    plt.xlabel("Agent")
    st.pyplot(plt.gcf())
    plt.clf()

# === Courbes Score vs Épisodes ===
if 'episodes' in filtered_df.columns:
    st.subheader("Courbes d’apprentissage (Score vs Épisodes)")
    fig = px.line(filtered_df, x="episodes", y="mean_score", color="agent", markers=True,
                  title="Score moyen en fonction du nombre d'épisodes")
    st.plotly_chart(fig, use_container_width=True)

# === Radar de performance ===
st.subheader("Radar de performance (normalisé)")
if not filtered_df.empty:
    radar_df = filtered_df.groupby("agent")[["mean_score", "mean_steps", "time"]].mean().reset_index()
    radar_norm = radar_df.copy()
    for col in ["mean_score", "mean_steps", "time"]:
        radar_norm[col] = (radar_norm[col] - radar_norm[col].min()) / (radar_norm[col].max() - radar_norm[col].min())

    fig = go.Figure()
    for _, row in radar_norm.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[["mean_score", "mean_steps", "time"]].tolist(),
            theta=["Score", "Steps", "Temps"],
            fill='toself',
            name=row["agent"]
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# === Scatter Score vs Steps ===
if 'mean_steps' in filtered_df.columns:
    st.subheader("Score vs Steps")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=filtered_df, x="mean_steps", y="mean_score", hue="agent", s=120)
    for _, row in filtered_df.iterrows():
        plt.text(row['mean_steps'] + 0.2, row['mean_score'], row['agent'], fontsize=9)
    plt.xlabel("Mean Steps")
    plt.ylabel("Mean Score")
    plt.title("Score vs Efficacité")
    st.pyplot(plt.gcf())
    plt.clf()

# === Heatmap des scores par environnement et agent (résumé global) ===
st.subheader("Heatmap des performances moyennes par environnement")

# Construction du pivot à partir de la feuille RésuméGlobal
heatmap_df = df_all.pivot_table(index='env', columns='agent', values='mean_score')

# Vérifier qu'il y a bien des données
if not heatmap_df.empty:
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Score Moyen'})
    plt.title("Performance des Agents par Environnement Secret")
    plt.ylabel("Environnement")
    plt.xlabel("Agent")
    st.pyplot(plt.gcf())
    plt.clf()
else:
    st.info("Aucune donnée disponible pour générer la heatmap.")

# === Détails des essais ===
st.subheader("Détails des expérimentations")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# === Meilleurs paramètres globaux ===
st.subheader("Meilleurs agents (global)")
st.dataframe(df_best[df_best["env"] == selected_env].reset_index(drop=True))
