import csv
import os
from datetime import datetime
from core.player_profile import get_player_path

def log_round(player_name, round_data):
    path = get_player_path(player_name)
    file_exists = os.path.isfile(path)

    with open(path, mode='a', newline='') as csvfile:
        fieldnames = ['timestamp', 'player', 'ai', 'result', 'agent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            'timestamp': datetime.now().isoformat(),
            'player': round_data['player'],
            'ai': round_data['ai'],
            'result': round_data['result'],
            'agent': round_data['agent']
        }
        writer.writerow(row)
