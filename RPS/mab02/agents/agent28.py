# Originally: signature_curve (SignatureCurveAgent)

import random

class Agent28:
    def __init__(self):
        self.name = "Agent28"

    def get_move(self, history):
        if len(history) < 6:
            return random.choice(['r', 'p', 's'])

        # Map moves to vectors: r = (1,0), p = (0,1), s = (-1,-1)
        mapping = {'r': (1, 0), 'p': (0, 1), 's': (-1, -1)}
        curve = [mapping[h['player']] for h in history[-6:]]
        dx = sum(p[0] for p in curve)
        dy = sum(p[1] for p in curve)

        if dx > dy and dx > 0:
            return 'p'  # assumuing r dominant, counter with p
        elif dy > dx and dy > 0:
            return 's'  # assumuing p dominant, counter with s
        else:
            return 'r'  # assumuing s dominant, counter with r