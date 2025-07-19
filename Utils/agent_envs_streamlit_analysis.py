import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧠 RL Experiments Dashboard")

# === Chargement des données ===
DATA_PATH = "../Reports/global_comparison.xlsx"


@st.cache_data
def load_data():
    xls = pd.ExcelFile(DATA_PATH)
    sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names if sheet not in ["RésuméGlobal", "BestParams"]}
    global_df = xls.parse("RésuméGlobal")
    best_df = xls.parse("BestParams")
    return sheets, global_df, best_df


sheets, df_all, df_best = load_data()

# === SIDEBAR : Paramètres ===
st.sidebar.title("🔧 Paramètres")
selected_env = st.sidebar.selectbox("🌍 Choisir un environnement", list(sheets.keys()))
df_env = sheets[selected_env]

agents = df_env["agent"].unique().tolist()
selected_agents = st.sidebar.multiselect("🤖 Sélectionner les agents", agents, default=agents)

filtered_df = df_env[df_env["agent"].isin(selected_agents)]

st.sidebar.markdown("---")
st.sidebar.markdown("🎚️ Filtrage des hyperparamètres")

hyperparams = ["gamma", "alpha", "epsilon", "theta", "planning_steps", "kappa", "episodes"]
for param in hyperparams:
    if param in filtered_df.columns:
        values = sorted(filtered_df[param].unique())
        if len(values) > 1:
            selected_values = st.sidebar.multiselect(f"{param}", values, default=values)
            filtered_df = filtered_df[filtered_df[param].isin(selected_values)]

# === MAIN : Résultats ===
st.markdown(f"### 📊 Résultats pour l'environnement `{selected_env}`")

# === Barplot des performances ===
st.subheader("🎯 Score moyen par agent")
plt.figure(figsize=(10, 5))
sns.barplot(data=filtered_df, x="agent", y="mean_score", ci="sd", palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("Score moyen")
plt.xlabel("Agent")
st.pyplot(plt.gcf())
plt.clf()

# === Barplot des steps moyens ===
if 'mean_steps' in filtered_df.columns:
    st.subheader("🚶 Steps moyens par épisode (efficacité)")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=filtered_df, x="agent", y="mean_steps", palette="mako")
    plt.xticks(rotation=45)
    plt.ylabel("Nombre moyen de steps")
    plt.xlabel("Agent")
    st.pyplot(plt.gcf())
    plt.clf()

# === Courbes d'apprentissage (score moyen vs episodes) ===
if 'episodes' in filtered_df.columns:
    st.subheader("📈 Courbes d’apprentissage (Score vs Épisodes)")
    fig = px.line(filtered_df, x="episodes", y="mean_score", color="agent", markers=True,
                  title="Score moyen en fonction du nombre d'épisodes")
    st.plotly_chart(fig, use_container_width=True)

# === Radar de performances (normalisées) ===
st.subheader("📊 Radar de performance des agents")
if not filtered_df.empty:
    radar_df = filtered_df.groupby("agent")[["mean_score", "mean_steps", "time"]].mean().reset_index()
    # Normalisation des valeurs
    radar_norm = radar_df.copy()
    for col in ["mean_score", "mean_steps", "time"]:
        radar_norm[col] = (radar_norm[col] - radar_norm[col].min()) / (radar_norm[col].max() - radar_norm[col].min())

    fig = go.Figure()
    for i, row in radar_norm.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[["mean_score", "mean_steps", "time"]].tolist(),
            theta=["Score", "Steps", "Temps"],
            fill='toself',
            name=row["agent"]
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# === Scatter score vs steps ===
if 'mean_steps' in filtered_df.columns:
    st.subheader("📉 Comparaison score vs steps moyens")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=filtered_df, x="mean_steps", y="mean_score", hue="agent", s=120)
    for i, row in filtered_df.iterrows():
        plt.text(row['mean_steps'] + 0.2, row['mean_score'], row['agent'], fontsize=9)
    plt.xlabel("Mean Steps")
    plt.ylabel("Mean Score")
    plt.title("Performance vs Efficacité")
    st.pyplot(plt.gcf())
    plt.clf()

# === Tableau détaillé ===
st.subheader("📄 Détails des essais")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# === Meilleurs paramètres ===
st.subheader("🏆 Meilleurs agents (global)")
st.dataframe(df_best[df_best["env"] == selected_env].reset_index(drop=True))
