# Originally: recurrence_depth (RecurrenceDepthAgent)

import random

class Agent29:
    def __init__(self):
        self.name = "Agent29"

    def get_move(self, history):
        if len(history) < 4:
            return random.choice(['r', 'p', 's'])

        moves = [h['player'] for h in history]
        best_pattern = None
        max_depth = 0

        for l in range(2, min(6, len(moves))):
            pattern = tuple(moves[-l:])
            for i in range(len(moves) - l):
                if tuple(moves[i:i + l]) == pattern:
                    if l > max_depth:
                        max_depth = l
                        best_pattern = pattern

        if best_pattern and len(best_pattern) > 0:
            last_move = best_pattern[-1]
            return self._counter(last_move)
        else:
            return random.choice(['r', 'p', 's'])

    def _counter(self, move):
        return {'r': 'p', 'p': 's', 's': 'r'}[move]