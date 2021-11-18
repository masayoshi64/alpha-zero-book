import pickle
import numpy as np
from tensorflow.keras import backend as K
from datetime import datetime

from tensorflow.python.keras.saving.save import load_model
from dual_network import DN_OUTPUT_SIZE
from pv_mcts import pv_mcts_scores
import os
from game import State  # nopep8


SP_GAME_COUNT = 500
SP_TEMPERATURE = 1.0


def first_player_value(ended_state: State):
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0


def write_data(history):
    now = datetime.now()
    os.makedirs("./data/", exist_ok=True)
    path = f"./data/{now.year:04}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}.history"
    with open(path, mode="wb") as f:
        pickle.dump(history, f)


def play(model):
    history = []
    state = State()
    while not state.is_done():
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])
        action = np.random.choice(state.legal_actions(), p=scores)
        state = state.next(action)
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value

    return history


def self_play():
    history = []
    model = load_model("./model/best.h5")
    for i in range(SP_GAME_COUNT):
        h = play(model)
        history.extend(h)
        print(f'\rSelfPlay {i+1}/{SP_GAME_COUNT}', end='')
    print()
    write_data(history)
    K.clear_session()
    del model


if __name__ == "__main__":
    self_play()
