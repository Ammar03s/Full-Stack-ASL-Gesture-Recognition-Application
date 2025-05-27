import random
from collections import defaultdict
import importlib

class MABController:
    def __init__(self, agent_names, epsilon=0.25):  # Reduced exploration rate
        self.epsilon = epsilon
        self.agent_names = agent_names
        self.agents = {name: self._load_agent(name) for name in agent_names}
        self.stats = defaultdict(lambda: {"wins": 0, "plays": 0})

    def _load_agent(self, agent_name):
        module = importlib.import_module(f"agents.{agent_name}")
        
        # Handle numbered agents (agent1, agent2, etc.)
        if agent_name.startswith('agent') and agent_name[5:].isdigit():
            class_name = agent_name.capitalize()  # agent1 -> Agent1
        else:
            # Legacy naming convention for older agents
            class_name = ''.join([part.capitalize() for part in agent_name.split('_')]) + "Agent"
            
        return getattr(module, class_name)()

    def select_agent(self):
        # With probability epsilon, explore
        if random.random() < self.epsilon:
            return random.choice(list(self.agents.values()))

        # Otherwise, exploit the best agent
        best_agent = None
        best_win_rate = -1

        for agent in self.agents.values():
            name = agent.name
            stats = self.stats[name]
            if stats["plays"] == 0:
                win_rate = 0
            else:
                # Calculate win rate as (total_plays - losses) / total_plays
                win_rate = (stats["plays"] - stats["wins"]) / stats["plays"]

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_agent = agent

        # If all agents untested, fallback to random
        return best_agent if best_agent else random.choice(list(self.agents.values()))

    def update_stats(self, agent_name, result):
        self.stats[agent_name]["plays"] += 1
        if result == "win":  # When player wins, agent loses
            self.stats[agent_name]["wins"] += 1

    def get_all_agents(self):
        return list(self.agents.values())

    def get_agent_by_name(self, name):
        return self.agents[name]
