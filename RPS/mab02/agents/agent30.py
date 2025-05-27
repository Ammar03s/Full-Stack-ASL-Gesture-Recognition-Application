# Originally: clustered_behavior (ClusteredBehaviorAgent)

import random
from collections import Counter





class Agent30:
    def __init__(self):
        self.name = "Agent30"

    def get_move(self, history):
        if len(history) < 6:
            return random.choice(['r', 'p', 's'])

        # Cluster behavior based on frequency in each third of the history
        third = len(history) // 3
        segments = [history[:third], history[third:2*third], history[2*third:]]
        cluster_scores = []

        for seg in segments:
            freq = Counter([h['player'] for h in seg])
            total = sum(freq.values())
            if total == 0:
                cluster_scores.append((0, ''))
                continue
            dominant = max(freq, key=freq.get)
            confidence = freq[dominant] / total
            cluster_scores.append((confidence, dominant))

        # Use the cluster with the highest confidence
        best = max(cluster_scores, key=lambda x: x[0])[1]
        return self._counter(best) if best else random.choice(['r', 'p', 's'])

    def _counter(self, move):
        return {'r': 'p', 'p': 's', 's': 'r'}[move]