from state import State, random_action


def mini_max(state):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    best_score = -float('inf')
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_score = score
    return best_score


def mini_max_action(state):
    best_action = 0
    best_score = -float('inf')
    str = ['', '']
    best_score = -float('inf')
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_score = score
            best_action = action
        str[0] = f"{str[0]}{action}, "
        str[1] = f"{str[1]}{score}, "
    print('action:', str[0], '\nscore: ', str[1], '\n')
    return best_action


def main():
    state = State()
    while not state.is_done():
        if state.is_first_player():
            action = mini_max_action(state)
        else:
            action = random_action(state)
        state = state.next(action)
        print(state)
        print()


if __name__ == "__main__":
    main()
