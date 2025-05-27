def evaluate(player_move, ai_move):
    beats = {
        'r': 's',  # rock beats scissors
        'p': 'r',  # paper beats rock
        's': 'p',  # scissors beats paper
    }

    if player_move == ai_move:
        return "draw"
    elif beats[player_move] == ai_move:
        return "win"
    else:
        return "lose"
