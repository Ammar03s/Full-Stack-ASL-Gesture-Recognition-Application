# Originally: reactionary (ReactionaryAgent)

import random

class Agent19:
    def __init__(self):
        self.name = "Agent19"

    def get_move(self, history):
        if not history:
            return random.choice(['r', 'p', 's'])

        last_opp = history[-1]['player']  # Player's last move
        last_me = history[-1]['ai']       # AI's last move
        score = self._get_score(last_me, last_opp)

        if score <= 1:  # lost or tied
            # Change to counter the opponent's last move
            counter_map = {'r': 'p', 'p': 's', 's': 'r'}
            return counter_map[last_opp]
        else:
            # Keep same move if it won
            return last_me

    def _get_score(self, my_move, opp_move):
        if my_move == opp_move:
            return 1
        elif (my_move == 'r' and opp_move == 's') or \
             (my_move == 'p' and opp_move == 'r') or \
             (my_move == 's' and opp_move == 'p'):
            return 2
        else:
            return 0
