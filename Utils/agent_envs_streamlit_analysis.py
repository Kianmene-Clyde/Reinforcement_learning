import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")

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

# === Sidebar ===
st.sidebar.title("🔧 Paramètres")
selected_env = st.sidebar.selectbox("Choisir un environnement", list(sheets.keys()))
df_env = sheets[selected_env]

agents = df_env["agent"].unique().tolist()
selected_agents = st.sidebar.multiselect("Filtrer les agents", agents, default=agents)

filtered_df = df_env[df_env["agent"].isin(selected_agents)]

# === Filtrage dynamique par hyperparamètres ===
st.sidebar.markdown("---")
st.sidebar.markdown("🎚️ Filtrage par hyperparamètres")

hyperparams = ["gamma", "alpha", "epsilon", "theta", "planning_steps", "kappa", "episodes"]
selected_hyperparams = {}

for param in hyperparams:
    if param in filtered_df.columns:
        values = sorted(filtered_df[param].unique())
        if len(values) > 1:
            selected_values = st.sidebar.multiselect(f"{param}", values, default=values)
            filtered_df = filtered_df[filtered_df[param].isin(selected_values)]

# === Main ===
st.title("RL Experiments Dashboard")
st.markdown(f"### Résultats pour l'environnement : `{selected_env}`")

# === Graphique des scores ===
st.subheader("Performance moyenne par agent")
plt.figure(figsize=(10, 5))
sns.barplot(data=filtered_df, x="agent", y="mean_score", ci="sd", palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("Score moyen")
plt.xlabel("Agent")
st.pyplot(plt.gcf())
plt.clf()

# === Graphique de stabilité ===
st.subheader("Stabilité des agents (écart-type des scores)")
plt.figure(figsize=(10, 5))
sns.barplot(data=filtered_df, x="agent", y="std_score", palette="coolwarm")
plt.xticks(rotation=45)
plt.ylabel("Écart-type")
plt.xlabel("Agent")
st.pyplot(plt.gcf())
plt.clf()

# === Tableau détaillé ===
st.subheader("Détails des essais")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# === Meilleurs paramètres ===
st.subheader("🔝 Meilleurs agents (tous environnements)")
st.dataframe(df_best[df_best["env"] == selected_env].reset_index(drop=True))
