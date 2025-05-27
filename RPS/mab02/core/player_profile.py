import os
import pandas as pd

DATA_DIR = "data/players" #for now

def get_player_path(player_name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{player_name.lower()}.csv")

def load_player_history(player_name):
    path = get_player_path(player_name)
    if os.path.exists(path):
        return pd.read_csv(path).to_dict('records')
    return []

def save_player_history(player_name, history):
    path = get_player_path(player_name)
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
