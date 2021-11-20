from alpha_beta import alpha_beta_action
from game import State, random_action, argmax, evaluate_algorithm_of


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


def main():
    next_actions = (mcs_action, random_action)
    evaluate_algorithm_of("VS_Random {:.3f}", next_actions, 100)
    next_actions = (mcs_action, alpha_beta_action)
    evaluate_algorithm_of("VS_AlphaBeta {:.3f}", next_actions, 100)


if __name__ == "__main__":
    main()
