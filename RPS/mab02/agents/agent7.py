# Originally: transition_matrix (TransitionMatrixAgent)

import random
from collections import defaultdict, Counter

class Agent7:
    def __init__(self):
        self.name = "Agent7"

    def get_move(self, history):
        if len(history) < 2:
            return random.choice(['r', 'p', 's'])

        matrix = defaultdict(Counter)
        for i in range(len(history) - 1):
            current = history[i]['player']
            next_ = history[i + 1]['player']
            matrix[current][next_] += 1

        last_move = history[-1]['player']
        if matrix[last_move]:
            prediction = matrix[last_move].most_common(1)[0][0]
        else:
            prediction = random.choice(['r', 'p', 's'])

        counter = {'r': 'p', 'p': 's', 's': 'r'}
        return counter[prediction]
