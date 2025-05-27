# Originally: copy_opponent (CopyOpponentAgent)

import random

class Agent17:
    def __init__(self):
        self.name = "Agent17"

    def get_move(self, history):
        if len(history) > 0:
            return history[-1]['player']
        else:
            return random.choice(['r', 'p', 's']) 