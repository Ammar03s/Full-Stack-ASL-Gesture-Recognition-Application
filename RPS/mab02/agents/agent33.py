# Originally: meta_vote (MetaVoteAgent)

import random
from collections import defaultdict, Counter
import math

# Import the numbered agents
from .agent7 import Agent7
from .agent16 import Agent16
from .agent26 import Agent26
from .agent37 import Agent37
from .agent38 import Agent38

class Agent33:
    def __init__(self):
        self.name = "Agent33"
        self.sub_agents = [Agent38(), Agent37(), Agent7(), Agent26(), Agent16()] 

    def get_move(self, history):
        votes = [agent.get_move(history) for agent in self.sub_agents]
        vote_counts = Counter(votes)
        majority_vote = vote_counts.most_common(1)[0][0]
        return majority_vote