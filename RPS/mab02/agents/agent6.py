# Originally: pattern_match (PatternMatchAgent)

import random

class Agent6:
    def __init__(self):
        self.name = "Agent6"
        self.default_pattern = ['r', 'p', 's']
        self.current_index = 0
        self.user_history = []
        self.min_history = 8 #adjustable
        self.pattern_lengths = [2, 3, 4, 5, 6]
        self.detected_pattern = None
        self.pattern_confidence = 0
        self.min_confidence = 1
        self.next_expected_moves = []
        self.debug = False

    def get_move(self, history):
        if history:
            self.user_history = [entry['player'] for entry in history]
            self._detect_pattern()

        if self.detected_pattern and self.pattern_confidence >= self.min_confidence:
            if self.next_expected_moves:
                predicted = self.next_expected_moves[0]
                return self._counter(predicted)

        if self.user_history:
            return self._counter(self.user_history[-1])

        # No history, return from default pattern
        move = self.default_pattern[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.default_pattern)
        return move

    def _counter(self, move):
        return {'r': 'p', 'p': 's', 's': 'r'}[move]

    def _detect_pattern(self):
        if len(self.user_history) < self.min_history:
            return False

        best_pattern = None
        best_confidence = 0
        best_repeats = 0

        for length in self.pattern_lengths:
            if len(self.user_history) < length * 2:
                continue

            for start in range(len(self.user_history) - length * 2 + 1):
                pattern = self.user_history[start:start + length]
                repeats = 0
                pos = start
                while pos + length <= len(self.user_history):
                    segment = self.user_history[pos:pos + length]
                    if segment == pattern:
                        repeats += 1
                        pos += length
                    else:
                        break

                if repeats >= 2 and repeats > best_repeats:
                    best_pattern = pattern
                    best_repeats = repeats
                    best_confidence = repeats

        if best_pattern and best_confidence >= self.min_confidence:
            self.detected_pattern = best_pattern
            self.pattern_confidence = best_confidence
            pos = len(self.user_history) % len(best_pattern)
            self.next_expected_moves = best_pattern[pos:] + best_pattern
            return True

        if self.pattern_confidence > 0:
            self.pattern_confidence -= 0.5

        if self.pattern_confidence < self.min_confidence:
            self.detected_pattern = None
            self.next_expected_moves = []

        return False