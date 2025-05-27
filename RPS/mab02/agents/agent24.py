# Originally: stochastic_transition_matrix (StochasticTransitionMatrixAgent)

import random
from collections import defaultdict

class Agent24:
    def __init__(self):
        self.name = "Agent24"
        self.move_map = {'r': 0, 'p': 1, 's': 2}
        self.reverse_map = {0: 'r', 1: 'p', 2: 's'}
        self.smoothing = 0.1  # Laplace smoothing

    def get_move(self, history):
        if len(history) < 2:
            return random.choice(['r', 'p', 's'])

        # Build transition matrix from history
        transition_matrix = defaultdict(lambda: [0, 0, 0])
        opponent_moves = [self.move_map[entry['player']] for entry in history]

        for i in range(len(opponent_moves) - 1):
            current_state = opponent_moves[i]
            next_state = opponent_moves[i + 1]
            transition_matrix[current_state][next_state] += 1

        # Predict based on last opponent move
        last_opponent_move = opponent_moves[-1]
        transitions = transition_matrix[last_opponent_move][:]

        # Apply Laplace smoothing
        for i in range(3):
            transitions[i] += self.smoothing

        total = sum(transitions)
        probabilities = [t / total for t in transitions]
        predicted_move = probabilities.index(max(probabilities))

        # 80% confidence in prediction
        if random.random() < 0.8:
            counter_move = (predicted_move + 1) % 3
            return self.reverse_map[counter_move]

        return random.choice(['r', 'p', 's'])
