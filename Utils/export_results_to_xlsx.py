import pandas as pd
from datetime import datetime
import os


def export_results(agent_name, env_name, stats, hyperparams, filename="results.xlsx"):
    data = {
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Agent": [agent_name],
        "Environnement": [env_name],
        **{f"Stat - {k}": [v] for k, v in stats.items()},
        **{f"Hyper - {k}": [v] for k, v in hyperparams.items()}
    }

    df = pd.DataFrame(data)

    if os.path.exists(filename):
        existing = pd.read_excel(filename)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_excel(filename, index=False)
    print(f"📁 Résultats exportés vers {filename}")
