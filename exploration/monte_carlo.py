from alpha_beta import alpha_beta_action
from state import State, random_action, argmax


def playout(state: State):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    return -playout(state.next(random_action(state)))


def mcs_action(state: State):
    legal_actions = state.legal_actions()
    values = [0] * len(legal_actions)
    for i, action in enumerate(legal_actions):
        for _ in range(10):
            values[i] += -playout(state.next(action))
    return legal_actions[argmax(values)]


EP_GAME_COUNT = 100


def first_player_point(ended_state):
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5


def play(next_actions):
    state = State()
    while not state.is_done():
        next_action = next_actions[0]\
            if state.is_first_player() else next_actions[1]
        action = next_action(state)
        state = state.next(action)
    return first_player_point(state)


def evaluate_algorithm_of(label, next_actions):
    total_point = 0
    for i in range(EP_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        print(f"\rEvaluate {i+1}/{EP_GAME_COUNT}")
    print('')

    average_point = total_point / EP_GAME_COUNT
    print(label.format(average_point))


def main():
    next_actions = (mcs_action, random_action)
    evaluate_algorithm_of("VS_Random {:.3f}", next_actions)
    next_actions = (mcs_action, alpha_beta_action)
    evaluate_algorithm_of("VS_AlphaBeta {:.3f}", next_actions)


if __name__ == "__main__":
    main()
