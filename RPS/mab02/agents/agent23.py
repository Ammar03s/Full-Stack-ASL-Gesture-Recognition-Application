# Originally: testing_pleasure (TestingPleasureAgent)

import random
from collections import defaultdict

class Agent23:
    def __init__(self):
        self.name = "Agent23"
        self.test_moves = ['r', 'p', 's']
        self.test_rounds = 9  # Total testing rounds

    def get_move(self, history):
        if len(history) < self.test_rounds:
            # Repeat each move 3 times
            return self.test_moves[len(history) // 3]

        # Analyze opponent responses during test phase
        responses = defaultdict(list)
        for i in range(min(self.test_rounds, len(history)) - 1):
            my_move = history[i]['ai']      # AI's previous move
            opp_move = history[i + 1]['player']  # Player's response to AI move
            responses[my_move].append(opp_move)

        # Evaluate most predictable opponent response
        best_move = 'r'
        best_confidence = 0
        for my_move, opp_responses in responses.items():
            if not opp_responses:
                continue
            most_common = max(set(opp_responses), key=opp_responses.count)
            confidence = opp_responses.count(most_common) / len(opp_responses)
            if confidence > best_confidence:
                best_confidence = confidence
                counter_map = {'r': 'p', 'p': 's', 's': 'r'}
                best_move = counter_map[most_common]

        # 10% random to stay fresh
        if random.random() < 0.1:
            return random.choice(['r', 'p', 's'])

        return best_move
