# Originally: markov_order2 (MarkovOrder2Agent)

import random
from collections import defaultdict, Counter

class Agent38:
    def __init__(self):
        self.name = "Agent38"

    def get_move(self, history):
        if len(history) < 3:
            return random.choice(['r', 'p', 's'])

        transition_counts = defaultdict(Counter)
        for i in range(len(history) - 2):
            prev2 = (history[i]['player'], history[i+1]['player'])
            next_move = history[i+2]['player']
            transition_counts[prev2][next_move] += 1

        last2 = (history[-2]['player'], history[-1]['player'])
        if last2 in transition_counts:
            predicted = transition_counts[last2].most_common(1)[0][0]
            return self._counter(predicted)

        return random.choice(['r', 'p', 's'])

    def _counter(self, move):
        return {'r': 'p', 'p': 's', 's': 'r'}[move]
