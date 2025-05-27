from core.mab_controller import MABController
from core.player_profile import load_player_history
from core.game_logger import log_round
from core.move_evaluator import evaluate
from core.stats_tracker import get_stats
from tabulate import tabulate
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualizer import plot_agent_usage, plot_win_loss_draw, plot_ai_winrate_trend, plot_agent_timeline, plot_best_agents_ranking

# List of all agent module names (38 agents with numbered naming)
AGENT_NAMES = [
    "agent1", "agent2", "agent3", "agent4", "agent5", "agent6", "agent7", "agent8", 
    "agent9", "agent10", "agent11", "agent12", "agent13", "agent14", "agent15", "agent16",
    "agent17", "agent18", "agent19", "agent20", "agent21", "agent22", "agent23", "agent24",
    "agent25", "agent26", "agent27", "agent28", "agent29", "agent30", "agent31", "agent32",
    "agent33", "agent34", "agent35", "agent36", "agent37", "agent38"
]

def display_score(player_wins, ai_wins, draws):
    print("\nCurrent Score:")
    print(f"Player: {player_wins} | AI: {ai_wins} | Draws: {draws}")

def main():
    print("Welcome!!")
    player_name = input("Enter your name: ").strip().lower()

    history = load_player_history(player_name)
    mab = MABController(agent_names=AGENT_NAMES)
    
    # Initialize score tracking
    player_wins = 0
    ai_wins = 0
    draws = 0

    print("\nEnter 'r' for Rock, 'p' for Paper, 's' for Scissors. Type 'q' to Quit.\n")

    while True:
        player_move = input("Your move (r/p/s) or q to Quit: ").strip().lower()
        if player_move == 'q':
            break
        if player_move not in ['r', 'p', 's']:
            print("Invalid input. Please enter r, p, s, q.")
            continue

        agent = mab.select_agent()
        ai_move = agent.get_move(history)
        result = evaluate(player_move, ai_move)

        # Update score
        if result == 'win':
            player_wins += 1
        elif result == 'lose':
            ai_wins += 1
        else:
            draws += 1

        print(f"\n🤖 AI ({agent.name}) played: {ai_move.upper()} → Result: {result.upper()}")
        display_score(player_wins, ai_wins, draws)

        round_data = {
            'player': player_move,
            'ai': ai_move,
            'result': result,
            'agent': agent.name
        }

        history.append(round_data)
        log_round(player_name, round_data)
        mab.update_stats(agent.name, result)

    # Show final stats in table format
    print("\nGame Summary:")
    stats = get_stats(player_name)
    
    # Overall stats table
    overall_table = [
        ["Rounds Played", stats['rounds']],
        ["Player Wins", stats['player_wins']],
        ["AI Wins", stats['ai_wins']],
        ["Draws", stats['draws']],
        ["Player Win Rate", f"{stats['win_rate']}%"]
    ]
    print(tabulate(overall_table, tablefmt="grid"))
    
    # Agent performance table
    print("\nAgent Performance:")
    agent_table = []
    for agent_name, agent_stats in stats["agent_stats"].items():
        agent_table.append([
            agent_name,
            agent_stats['rounds'],
            agent_stats['wins'],
            agent_stats['losses'],
            agent_stats['draws'],
            f"{agent_stats['win_rate']}%"
        ])
    
    print(tabulate(agent_table, 
                  headers=["Agent", "Rounds", "Player Wins", "AI Wins", "Draws", "Player Win Rate"],
                  tablefmt="grid"))

    print("\nThank you for playing 👋👋")
    
    # Show AI performance visualizations
    print("\nGenerating AI performance visualizations...")
    plot_ai_winrate_trend(player_name)
    plot_agent_timeline(player_name)
    plot_best_agents_ranking(player_name)
    plot_win_loss_draw(player_name)

if __name__ == "__main__":
    main()
