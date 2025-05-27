import pandas as pd
from core.player_profile import get_player_path

def get_stats(player_name):
    path = get_player_path(player_name)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {
            "rounds": 0,
            "player_wins": 0,
            "ai_wins": 0,
            "draws": 0,
            "win_rate": 0.0,
            "agent_stats": {}
        }

    total = len(df)
    player_wins = (df['result'] == 'win').sum()
    ai_wins = (df['result'] == 'lose').sum()
    draws = (df['result'] == 'draw').sum()

    # Win rate excluding draws
    non_draws = total - draws
    win_rate = player_wins / non_draws if non_draws > 0 else 0.0

    # Calculate agent-specific stats
    agent_stats = {}
    for agent in df['agent'].unique():
        agent_df = df[df['agent'] == agent]
        agent_total = len(agent_df)
        agent_wins = (agent_df['result'] == 'win').sum()
        agent_losses = (agent_df['result'] == 'lose').sum()
        agent_draws = (agent_df['result'] == 'draw').sum()
        agent_non_draws = agent_total - agent_draws
        agent_win_rate = agent_wins / agent_non_draws if agent_non_draws > 0 else 0.0
        
        agent_stats[agent] = {
            "rounds": agent_total,
            "wins": agent_wins,
            "losses": agent_losses,
            "draws": agent_draws,
            "win_rate": round(agent_win_rate * 100, 2)
        }

    return {
        "rounds": total,
        "player_wins": player_wins,
        "ai_wins": ai_wins,
        "draws": draws,
        "win_rate": round(win_rate * 100, 2),
        "agent_stats": agent_stats
    }
