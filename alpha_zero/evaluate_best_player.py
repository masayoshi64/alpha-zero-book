from game import random_action, alpha_beta_action, mcts_action
from pv_mcts import get_pv_mcts_action
from tensorflow.keras.models import load_model
from evaluate_network import play


EP_GAME_COUNT = 10


def evaluate_algorithm_of(label, next_actions, game_count):
    total_point = 0
    for i in range(game_count):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play((next_actions[1], next_actions[0]))

        print(f"Evaluate {i+1}/{game_count}\r", end='')
    print('')

    average_point = total_point / game_count
    print(label, average_point)


def evaluate_best_player():
    model = load_model('./model/best.h5')
    next_pv_mcts_action = get_pv_mcts_action(model, 0.0)

    next_actions = (next_pv_mcts_action, random_action)
    evaluate_algorithm_of('VS_Random', next_actions, EP_GAME_COUNT)

    next_actions = (next_pv_mcts_action, alpha_beta_action)
    evaluate_algorithm_of('VS_AlphaBeta', next_actions, EP_GAME_COUNT)

    next_actions = (next_pv_mcts_action, mcts_action)
    evaluate_algorithm_of('VS_MCTS', next_actions, EP_GAME_COUNT)


if __name__ == "__main__":
    evaluate_best_player()
