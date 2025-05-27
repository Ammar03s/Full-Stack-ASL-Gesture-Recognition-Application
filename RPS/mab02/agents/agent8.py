# Originally: decay_weighted_freq (DecayWeightedFreqAgent)

import random
from collections import defaultdict

class Agent8:
    def __init__(self, decay=0.9):
        self.name = "Agent8"
        self.decay = decay

    def get_move(self, history):
        if not history:
            return random.choice(['r', 'p', 's'])

        scores = defaultdict(float)
        weight = 1.0

        for move in reversed([entry['player'] for entry in history]):
            scores[move] += weight
            weight *= self.decay

        predicted = max(scores, key=scores.get)
        counter = {'r': 'p', 'p': 's', 's': 'r'}
        return counter[predicted]
