# Originally: frequency_decay_sigmoid (FrequencyDecaySigmoidAgent)

import random
class Agent36:
    def __init__(self):
        self.name = "Agent36"

    def get_move(self, history):
        if not history:
            return random.choice(['r', 'p', 's'])

        score = {'r': 0, 'p': 0, 's': 0}
        n = len(history)
        for i, h in enumerate(reversed(history)):
            weight = 1 / (1 + pow(2.718, -(n - i)))  # sigmoid-like weight
            score[h['player']] += weight

        predicted = max(score, key=score.get)
        return self._counter(predicted)

    def _counter(self, move):
        return {'r': 'p', 'p': 's', 's': 'r'}[move]