# Originally: gradient_signal (GradientSignalAgent)

import random
class Agent32:
    def __init__(self):
        self.name = "Agent32"

    def get_move(self, history):
        if len(history) < 4:
            return random.choice(['r', 'p', 's'])

        mapping = {'r': 0, 'p': 1, 's': 2}
        grad = 0
        for i in range(-3, 0):
            a = mapping[history[i]['player']]
            b = mapping[history[i + 1]['player']]
            grad += b - a

        prediction = (mapping[history[-1]['player']] + (1 if grad > 0 else -1)) % 3
        reverse = {0: 'r', 1: 'p', 2: 's'}
        counter = {'r': 'p', 'p': 's', 's': 'r'}
        return counter[reverse[prediction]]