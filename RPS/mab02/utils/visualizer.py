import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.player_profile import get_player_path

def plot_combined_dashboard(player_name):
    """Create a combined dashboard with all visualizations in one figure"""
    path = get_player_path(player_name)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("No gameplay data found.")
        return

    if df.empty:
        print("No game data to plot.")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Rock-Paper-Scissors AI Performance Dashboard - {player_name.title()}', 
                 fontsize=20, fontweight='bold', y=0.95)

    # 1. AI Winrate Trend (Top Left)
    non_draw_games = df[df['result'] != 'draw'].copy()
    if not non_draw_games.empty:
        non_draw_games['ai_win'] = (non_draw_games['result'] == 'lose').astype(int)
        non_draw_games['cumulative_ai_wins'] = non_draw_games['ai_win'].cumsum()
        non_draw_games['game_number'] = range(1, len(non_draw_games) + 1)
        non_draw_games['ai_winrate'] = non_draw_games['cumulative_ai_wins'] / non_draw_games['game_number'] * 100

        ax1.plot(non_draw_games['game_number'], non_draw_games['ai_winrate'], 
                 linewidth=3, color='#E74C3C', marker='o', markersize=6)
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% (Random)')
        ax1.set_title('AI Winrate Trend Over Games', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Game Number (Excluding Draws)', fontsize=12)
        ax1.set_ylabel('AI Winrate (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)

    # 2. Agent Timeline (Top Right)
    unique_agents = sorted(df['agent'].unique())
    agent_positions = {agent: i for i, agent in enumerate(unique_agents)}
    game_numbers = range(1, len(df) + 1)
    agent_y_positions = [agent_positions[agent] for agent in df['agent']]
    colors = {'win': '#4CAF50', 'lose': '#F44336', 'draw': '#FFC107'}
    point_colors = [colors[result] for result in df['result']]

    ax2.scatter(game_numbers, agent_y_positions, c=point_colors, alpha=0.8, s=40, edgecolors='black', linewidth=0.5)
    ax2.set_title('Agent Usage Timeline', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Game Number', fontsize=12)
    ax2.set_ylabel('Agents', fontsize=12)
    ax2.set_yticks(range(len(unique_agents)))
    ax2.set_yticklabels(unique_agents, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.grid(True, alpha=0.2, axis='y')

    # 3. Best Agents Ranking (Bottom Left)
    agent_stats = df.groupby('agent').agg({
        'result': ['count', lambda x: (x == 'lose').sum(), lambda x: (x == 'draw').sum()]
    }).round(2)
    
    agent_stats.columns = ['total_games', 'ai_wins', 'draws']
    agent_stats['decisive_games'] = agent_stats['total_games'] - agent_stats['draws']
    agent_stats['ai_winrate'] = ((agent_stats['ai_wins'] / agent_stats['decisive_games']) * 100).fillna(0).round(1)
    
    qualified_agents = agent_stats[agent_stats['decisive_games'] >= 2].copy()
    if qualified_agents.empty:
        qualified_agents = agent_stats.copy()
    
    qualified_agents = qualified_agents.sort_values(['ai_winrate', 'total_games'], ascending=[False, False])
    top_agents = qualified_agents.head(15)  # Show top 15 for readability

    y_pos = range(len(top_agents))
    bars = ax3.barh(y_pos, top_agents['ai_winrate'], color='#E74C3C', alpha=0.8, edgecolor='black')
    
    for i, (bar, winrate, games) in enumerate(zip(bars, top_agents['ai_winrate'], top_agents['total_games'])):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{winrate:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_agents.index, fontsize=10)
    ax3.set_xlabel('AI Winrate (%)', fontsize=12)
    ax3.set_title('Top Performing Agents', fontsize=14, fontweight='bold')
    ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlim(0, max(100, top_agents['ai_winrate'].max() + 10))
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Overall Results Pie Chart (Bottom Right)
    result_counts = df['result'].value_counts()
    ai_wins = result_counts.get('lose', 0)
    player_wins = result_counts.get('win', 0)
    draws = result_counts.get('draw', 0)
    
    total_decisive = ai_wins + player_wins
    ai_winrate = (ai_wins / total_decisive * 100) if total_decisive > 0 else 0
    
    sizes = [ai_wins, player_wins, draws] if draws > 0 else [ai_wins, player_wins]
    labels = [f'AI Wins ({ai_wins})', f'Player Wins ({player_wins})']
    colors_pie = ['#E74C3C', '#4CAF50']
    
    if draws > 0:
        labels.append(f'Draws ({draws})')
        colors_pie.append('#FFC107')

    ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie)
    ax4.set_title(f'Overall Results\nAI Winrate: {ai_winrate:.1f}% (excl. draws)', 
                  fontsize=14, fontweight='bold')

    # Add legend for timeline colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='Player Win'),
                      Patch(facecolor='#F44336', label='AI Win'),
                      Patch(facecolor='#FFC107', label='Draw')]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    
    # Save the combined dashboard
    filename = f'rps_dashboard_{player_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Combined dashboard saved as: {filename}")
    plt.show()

def plot_ai_winrate_trend(player_name):
    """Plot AI winrate trend over games (excluding draws)"""
    path = get_player_path(player_name)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("No gameplay data found.")
        return

    if df.empty:
        print("No game data to plot.")
        return

    # Calculate cumulative AI winrate (excluding draws)
    non_draw_games = df[df['result'] != 'draw'].copy()
    if non_draw_games.empty:
        print("No decisive games to analyze.")
        return

    # AI wins when result is 'lose' (player loses)
    non_draw_games['ai_win'] = (non_draw_games['result'] == 'lose').astype(int)
    non_draw_games['cumulative_ai_wins'] = non_draw_games['ai_win'].cumsum()
    non_draw_games['game_number'] = range(1, len(non_draw_games) + 1)
    non_draw_games['ai_winrate'] = non_draw_games['cumulative_ai_wins'] / non_draw_games['game_number'] * 100

    plt.figure(figsize=(12, 6))
    plt.plot(non_draw_games['game_number'], non_draw_games['ai_winrate'], 
             linewidth=2, color='#E74C3C', marker='o', markersize=4)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% (Random)')
    plt.title(f'AI Winrate Trend Over Games - {player_name.title()}', fontsize=14, fontweight='bold')
    plt.xlabel('Game Number (Excluding Draws)', fontsize=12)
    plt.ylabel('AI Winrate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

def plot_agent_timeline(player_name):
    """Dots graph: Agent names on Y-axis, game numbers on X-axis, dots show which agent was used"""
    path = get_player_path(player_name)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("No gameplay data found.")
        return

    if df.empty:
        print("No agent data to plot.")
        return

    # Get unique agents and assign them y-positions
    unique_agents = sorted(df['agent'].unique())
    agent_positions = {agent: i for i, agent in enumerate(unique_agents)}
    
    # Create the plot data
    game_numbers = range(1, len(df) + 1)
    agent_y_positions = [agent_positions[agent] for agent in df['agent']]
    
    # Color code by result
    colors = {'win': '#4CAF50', 'lose': '#F44336', 'draw': '#FFC107'}
    point_colors = [colors[result] for result in df['result']]

    plt.figure(figsize=(15, max(8, len(unique_agents) * 0.6)))
    plt.scatter(game_numbers, agent_y_positions, c=point_colors, alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    
    plt.title(f'Agent Usage by Game - {player_name.title()}', fontsize=16, fontweight='bold')
    plt.xlabel('Game Number', fontsize=14)
    plt.ylabel('Agents', fontsize=14)
    plt.yticks(range(len(unique_agents)), unique_agents, fontsize=11)
    plt.grid(True, alpha=0.3, axis='x')
    plt.grid(True, alpha=0.2, axis='y')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='Player Win'),
                      Patch(facecolor='#F44336', label='AI Win'),
                      Patch(facecolor='#FFC107', label='Draw')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set x-axis to show every 5th game number for readability
    if len(df) > 20:
        plt.xticks(range(0, len(df)+1, max(1, len(df)//10)))
    
    plt.tight_layout()
    plt.show()

def plot_best_agents_ranking(player_name):
    """Best agents sorted by AI winrate and performance metrics"""
    path = get_player_path(player_name)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("No gameplay data found.")
        return

    if df.empty:
        print("No agent data to plot.")
        return

    # Calculate agent statistics
    agent_stats = df.groupby('agent').agg({
        'result': ['count', lambda x: (x == 'lose').sum(), lambda x: (x == 'draw').sum()]
    }).round(2)
    
    agent_stats.columns = ['total_games', 'ai_wins', 'draws']
    agent_stats['decisive_games'] = agent_stats['total_games'] - agent_stats['draws']
    agent_stats['ai_winrate'] = ((agent_stats['ai_wins'] / agent_stats['decisive_games']) * 100).fillna(0).round(1)
    
    # Filter agents with at least 2 decisive games for fair comparison
    qualified_agents = agent_stats[agent_stats['decisive_games'] >= 2].copy()
    
    if qualified_agents.empty:
        # If no agents have enough games, show all
        qualified_agents = agent_stats.copy()
    
    # Sort by AI winrate (descending) then by total games (descending)
    qualified_agents = qualified_agents.sort_values(['ai_winrate', 'total_games'], ascending=[False, False])

    plt.figure(figsize=(14, max(8, len(qualified_agents) * 0.4)))
    
    # Create horizontal bar chart
    y_pos = range(len(qualified_agents))
    bars = plt.barh(y_pos, qualified_agents['ai_winrate'], color='#E74C3C', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, winrate, games) in enumerate(zip(bars, qualified_agents['ai_winrate'], qualified_agents['total_games'])):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{winrate:.1f}% ({games} games)', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.yticks(y_pos, qualified_agents.index, fontsize=11)
    plt.xlabel('AI Winrate (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Agents', fontsize=14, fontweight='bold')
    plt.title(f'Best Performing Agents Ranking - {player_name.title()}', fontsize=16, fontweight='bold')
    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='50% (Random)')
    plt.xlim(0, max(100, qualified_agents['ai_winrate'].max() + 10))
    plt.grid(True, alpha=0.3, axis='x')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Keep these for backward compatibility but make them simple
def plot_agent_usage(player_name):
    """Simple agent usage count"""
    plot_best_agents_ranking(player_name)  # Redirect to best agents ranking

def plot_win_loss_draw(player_name):
    """Simple overall stats"""
    path = get_player_path(player_name)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("No gameplay data found.")
        return

    if df.empty:
        print("No game data to plot.")
        return

    result_counts = df['result'].value_counts()
    
    # AI perspective: AI wins when player loses
    ai_wins = result_counts.get('lose', 0)
    player_wins = result_counts.get('win', 0)
    #draws = result_counts.get('draw', 0)
    
    total_decisive = ai_wins + player_wins
    ai_winrate = (ai_wins / total_decisive * 100) if total_decisive > 0 else 0
    
    sizes = [ai_wins, player_wins]
    labels = [f'AI Wins ({ai_wins})', f'Player Wins ({player_wins})']
    colors = ['#E74C3C', '#4CAF50']

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f'Overall Results - {player_name.title()}\nAI Winrate: {ai_winrate:.1f}% (excluding draws)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
