from game.ko_tron import KOTron


def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_terminal():
        return state.evaluate()

    if maximizing_player:
        max_eval = -float('inf')
        for move in state.get_legal_moves():
            eval = minimax(state.make_move(move), depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for move in state.get_legal_moves():
            eval = minimax(state.make_move(move), depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_terminal():
        return state.evaluate()

    if maximizing_player:
        max_eval = -float('inf')
        for move in state.get_legal_moves():
            eval = minimax(state.make_move(move), depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for move in state.get_legal_moves():
            eval = minimax(state.make_move(move), depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval

def find_best_move(game: KOTron, player_index, depth=5):


    best_move = None
    best_value = -float('inf') if state.current_player == 'max' else float('inf')

    possible_directions = KOTron.get_possible_directions(game, player_index)

    for direction in possible_directions:

        new_state = KOTron.next(game, )
        move_value = minimax(new_state, depth-1, -float('inf'), float('inf'), state.current_player != 'max')

        if state.current_player == 'max' and move_value > best_value:
            best_value = move_value
            best_move = move
        elif state.current_player != 'max' and move_value < best_value:
            best_value = move_value
            best_move = move

    return best_move
