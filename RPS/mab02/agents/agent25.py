# Originally: opponent_transition_matrix (OpponentTransitionMatrixAgent)

import random
from collections import defaultdict

class Agent25:
    def __init__(self):
        self.name = "Agent25"

    def get_move(self, history):
        if len(history) < 2:
            return random.choice(['r', 'p', 's'])

        response_matrix = defaultdict(lambda: {'r': 0, 'p': 0, 's': 0})

        # Reconstruct response matrix from full history
        for entry in history:
            our_move = entry['ai']      # AI's previous move
            opp_move = entry['player']  # Player's move in response
            response_matrix[our_move][opp_move] += 1

        # Predict best move based on expected score
        best_move = 'r'
        best_expected_score = float('-inf')

        for potential_move in ['r', 'p', 's']:
            responses = response_matrix[potential_move]
            total = sum(responses.values())
            if total == 0:
                continue

            expected_score = 0
            for opp_move, count in responses.items():
                probability = count / total
                expected_score += probability * self._get_score(potential_move, opp_move)

            if expected_score > best_expected_score:
                best_expected_score = expected_score
                best_move = potential_move

        return best_move

    def _get_score(self, my_move, opp_move):
        if my_move == opp_move:
            return 0
        elif (my_move == 'r' and opp_move == 's') or \
             (my_move == 'p' and opp_move == 'r') or \
             (my_move == 's' and opp_move == 'p'):
            return 1
        else:
            return -1
