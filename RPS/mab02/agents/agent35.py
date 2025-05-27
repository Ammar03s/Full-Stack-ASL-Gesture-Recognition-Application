# Originally: entropy_hunter (EntropyHunterAgent)

import random
from collections import Counter
import math

class Agent35:
    def __init__(self):
        self.name = "Agent35"

    def get_move(self, history):
        if len(history) < 5:
            return random.choice(['r', 'p', 's'])

        segments = [history[i:i+3] for i in range(len(history)-2)]
        min_entropy = float('inf')
        best_move = None

        for seg in segments:
            counter = Counter([entry['player'] for entry in seg])
            total = sum(counter.values())
            entropy = -sum((v/total) * math.log2(v/total) for v in counter.values())
            if entropy < min_entropy:
                min_entropy = entropy
                best_move = seg[-1]['player']

        return {'r': 'p', 'p': 's', 's': 'r'}[best_move] if best_move else random.choice(['r', 'p', 's'])