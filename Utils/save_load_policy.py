import pickle
import os


def save_policy(policy, filename, directory="saved_policies"):
    """
    Sauvegarde une politique (policy) dans un fichier .pkl
    """
    path = os.path.join(directory, filename)

    # Crée tous les dossiers parents si besoin
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(policy, f)
    print(f"Politique sauvegardée dans {path}")


def load_policy(filename, directory="saved_policies"):
    """
    Charge une politique (policy) depuis un fichier .pkl
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, "rb") as f:
        policy = pickle.load(f)
    print(f"Politique chargée depuis {path}")
    return policy
