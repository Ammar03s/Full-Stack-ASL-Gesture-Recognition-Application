# Originally: markov_chain (MarkovChainAgent)

import random
from collections import defaultdict, Counter

class Agent5:
    def __init__(self):
        self.name = "Agent5"
        self.transition_table = defaultdict(Counter)

    def get_move(self, history):
        if len(history) < 2:
            return random.choice(['r', 'p', 's'])

        # Build transition table from history
        for i in range(len(history) - 1):
            prev = history[i]['player']
            next_ = history[i + 1]['player']
            self.transition_table[prev][next_] += 1

        last_move = history[-1]['player']
        predictions = self.transition_table[last_move]

        if predictions:
            predicted = predictions.most_common(1)[0][0]
        else:
            predicted = random.choice(['r', 'p', 's'])

        # Counter move
        counter_moves = {'r': 'p', 'p': 's', 's': 'r'}
        return counter_moves[predicted]
