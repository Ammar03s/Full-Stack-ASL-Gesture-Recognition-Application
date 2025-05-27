# Originally: frequency_counter (FrequencyCounterAgent)

import random
from collections import Counter

class Agent4:
    def __init__(self):
        self.name = "Agent4"

    def get_move(self, history):
        if not history:
            return random.choice(['r', 'p', 's'])

        player_moves = [entry['player'] for entry in history]
        most_common = Counter(player_moves).most_common(1)[0][0]

        # Counter move: r → p, p → s, s → r
        counter_moves = {'r': 'p', 'p': 's', 's': 'r'}
        return counter_moves[most_common]
