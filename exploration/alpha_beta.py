from game import State, random_action


def alpha_beta(state, alpha, beta):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score
        if alpha >= beta:
            return alpha
    return alpha


def alpha_beta_action(state):
    best_action = 0
    best_score = -float('inf')
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def main():
    state = State()
    while not state.is_done():
        if state.is_first_player():
            action = alpha_beta_action(state)
        else:
            action = random_action(state)
        state = state.next(action)
        print(state)
        print()


if __name__ == "__main__":
    main()
