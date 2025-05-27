# Originally: counter_reactionary (CounterReactionaryAgent)

import random

class Agent20:
    def __init__(self):
        self.name = "Agent20"
        self.last_action = None

    def get_move(self, history):
        if len(history) == 0:
            self.last_action = random.choice(['r', 'p', 's'])
        elif len(history) > 0:
            last_opp = history[-1]['player']
            score = self._get_score(self.last_action, last_opp)
            
            if score == 1:  # tied, change strategy
                # Move to what would beat what beats the opponent's last move
                beat_opponent = {'r': 'p', 'p': 's', 's': 'r'}[last_opp]
                self.last_action = {'r': 'p', 'p': 's', 's': 'r'}[beat_opponent]
            else:
                # Counter the opponent's last move
                self.last_action = {'r': 'p', 'p': 's', 's': 'r'}[last_opp]
        
        return self.last_action

    def _get_score(self, my_move, opp_move):
        """Returns: 2 for win, 1 for tie, 0 for loss"""
        if my_move == opp_move:
            return 1
        elif (my_move == 'r' and opp_move == 's') or \
             (my_move == 'p' and opp_move == 'r') or \
             (my_move == 's' and opp_move == 'p'):
            return 2
        else:
            return 0 