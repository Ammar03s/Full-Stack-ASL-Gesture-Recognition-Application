#!/usr/bin/env python3

from core.mab_controller import MABController
from interface.play_typing import AGENT_NAMES

def test_all_agents():
    print(f"Testing {len(AGENT_NAMES)} agents...")
    
    try:
        mab = MABController(AGENT_NAMES)
        print("✅ SUCCESS! All agents loaded successfully!")
        print(f"Total agents loaded: {len(mab.get_all_agents())}")
        
        # Test with realistic game history
        test_history = [
            {'player': 'r', 'ai': 'p', 'result': 'lose', 'agent': 'agent1'},
            {'player': 'p', 'ai': 's', 'result': 'lose', 'agent': 'agent2'},
            {'player': 's', 'ai': 'r', 'result': 'lose', 'agent': 'agent3'},
            {'player': 'r', 'ai': 'r', 'result': 'draw', 'agent': 'agent4'},
            {'player': 'p', 'ai': 'p', 'result': 'draw', 'agent': 'agent5'},
            {'player': 's', 'ai': 'p', 'result': 'win', 'agent': 'agent6'},
            {'player': 'r', 'ai': 's', 'result': 'win', 'agent': 'agent7'},
            {'player': 'p', 'ai': 'r', 'result': 'win', 'agent': 'agent8'},
        ]
        
        print("\nTesting agent moves with realistic history...")
        failed_agents = []
        
        for agent_name in AGENT_NAMES:
            try:
                agent = mab.agents[agent_name]
                move = agent.get_move(test_history)
                if move in ['r', 'p', 's']:
                    print(f"  ✅ {agent_name}: {move}")
                else:
                    print(f"  ❌ {agent_name}: Invalid move '{move}'")
                    failed_agents.append(agent_name)
            except Exception as e:
                print(f"  ❌ {agent_name}: ERROR - {e}")
                failed_agents.append(agent_name)
        
        if failed_agents:
            print(f"\n❌ {len(failed_agents)} agents failed: {failed_agents}")
            return False
        else:
            print(f"\n🎉 All {len(AGENT_NAMES)} agents passed comprehensive testing!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_all_agents() 