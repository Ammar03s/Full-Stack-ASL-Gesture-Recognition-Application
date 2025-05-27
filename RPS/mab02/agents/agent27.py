# Originally: de_bruijn (DeBruijnAgent)

import random
from collections import defaultdict

class Agent27:
    def __init__(self):
        self.name = "Agent27"
        self.sequence_length = 4
        self.move_map = {'r': 0, 'p': 1, 's': 2}
        self.reverse_map = {0: 'r', 1: 'p', 2: 's'}

    def get_move(self, history):
        if len(history) < self.sequence_length:
            return random.choice(['r', 'p', 's'])

        opponent_moves = [self.move_map[entry['player']] for entry in history]
        patterns = defaultdict(list)

        for i in range(len(opponent_moves) - self.sequence_length + 1):
            sub_pattern = tuple(opponent_moves[i:i + self.sequence_length - 1])
            next_move = opponent_moves[i + self.sequence_length - 1]
            patterns[sub_pattern].append(next_move)

        current_pattern = tuple(opponent_moves[-(self.sequence_length - 1):])
        if current_pattern in patterns:
            recent_moves = patterns[current_pattern][-min(5, len(patterns[current_pattern])):]
            prediction = max(set(recent_moves), key=recent_moves.count)
            counter_move = (prediction + 1) % 3
            return self.reverse_map[counter_move]

        return random.choice(['r', 'p', 's'])
