# Originally: xgboost (XgboostAgent)

import random
from collections import defaultdict

class Agent26:
    def __init__(self):
        self.name = "Agent26"
        self.pattern_length = 5
        self.min_pattern_count = 3

    def get_move(self, history):
        if len(history) < self.pattern_length:
            return random.choice(['r', 'p', 's'])

        opponent_moves = [entry['player'] for entry in history]
        patterns = defaultdict(list)

        # Build pattern database from history
        for i in range(len(opponent_moves) - self.pattern_length + 1):
            sub_pattern = tuple(opponent_moves[i:i + self.pattern_length - 1])
            next_move = opponent_moves[i + self.pattern_length - 1]
            patterns[sub_pattern].append(next_move)

        # Match current pattern
        pattern = tuple(opponent_moves[-(self.pattern_length - 1):])
        if pattern in patterns and len(patterns[pattern]) >= self.min_pattern_count:
            next_moves = patterns[pattern]
            prediction = max(set(next_moves), key=next_moves.count)
            counter_map = {'r': 'p', 'p': 's', 's': 'r'}
            return counter_map[prediction]

        return random.choice(['r', 'p', 's'])
