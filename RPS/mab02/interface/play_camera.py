import cv2
import mediapipe as mp
import time
from tabulate import tabulate

from core.mab_controller import MABController
from core.player_profile import load_player_history
from core.game_logger import log_round
from core.move_evaluator import evaluate
from core.stats_tracker import get_stats
from utils.visualizer import plot_agent_usage, plot_win_loss_draw, plot_ai_winrate_trend, plot_agent_timeline, plot_best_agents_ranking

# List of agent modules (38 agents with numbered naming)
AGENT_NAMES = [
    "agent1", "agent2", "agent3", "agent4", "agent5", "agent6", "agent7", "agent8", 
    "agent9", "agent10", "agent11", "agent12", "agent13", "agent14", "agent15", "agent16",
    "agent17", "agent18", "agent19", "agent20", "agent21", "agent22", "agent23", "agent24",
    "agent25", "agent26", "agent27", "agent28", "agent29", "agent30", "agent31", "agent32",
    "agent33", "agent34", "agent35", "agent36", "agent37", "agent38"
]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_finger_count(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers_up = 0

    # Thumb (check x-axis due to rotation)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers_up += 1

    # Other 4 fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers_up += 1

    return fingers_up

def map_fingers_to_move(fingers_up):
    if fingers_up == 0:
        return 'r'  # rock
    elif fingers_up in [2, 3]:
        return 's'  # scissors
    elif fingers_up == 5:
        return 'p'  # paper
    return None

def display_score(player_wins, ai_wins, draws):
    return f"Score - Player: {player_wins} | AI: {ai_wins} | Draws: {draws}"

def main():
    print("🖐️ Rock-Paper-Scissors with MediaPipe")
    player_name = input("Enter your name: ").strip().lower()

    history = load_player_history(player_name)
    mab = MABController(agent_names=AGENT_NAMES)
    
    # Initialize score tracking
    player_wins = 0
    ai_wins = 0
    draws = 0

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        print("\nPress 'q' to quit.")
        move_cooldown = 2  # seconds
        last_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            player_move = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    fingers_up = get_finger_count(hand_landmarks)
                    move = map_fingers_to_move(fingers_up)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if move and time.time() - last_time > move_cooldown:
                        player_move = move
                        last_time = time.time()

                        # Play round
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

                        print(f"\n🧍You: {player_move.upper()}  | 🤖 AI ({agent.name}): {ai_move.upper()} → {result.upper()}")
                        print(display_score(player_wins, ai_wins, draws))

                        round_data = {
                            'player': player_move,
                            'ai': ai_move,
                            'result': result,
                            'agent': agent.name
                        }

                        history.append(round_data)
                        log_round(player_name, round_data)
                        mab.update_stats(agent.name, result)

            # Display instructions and score
            cv2.putText(frame, "Show Rock (0), Scissors (2,3), Paper (5)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, display_score(player_wins, ai_wins, draws), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Rock-Paper-Scissors - Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

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

    print("\nThanks for playing with gestures! 👋")
    
    # Show AI performance visualizations
    print("\nGenerating AI performance visualizations...")
    plot_ai_winrate_trend(player_name)
    plot_agent_timeline(player_name)
    plot_best_agents_ranking(player_name)
    plot_win_loss_draw(player_name)

if __name__ == "__main__":
    main()
