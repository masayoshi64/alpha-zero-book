import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from math import sqrt
from dual_network import DN_INPUT_SHAPE
from game import State


PV_EVALUATE_COUNT = 50


def predict(model: Model, state: State):
    a, b, c = DN_INPUT_SHAPE
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)

    y = model.predict(x, batch_size=1)
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1

    value = y[1][0][0]
    return policies, value


def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores


class Node:
    def __init__(self, state: State, p: float) -> None:
        self.state = state
        self.p = p
        self.w = 0
        self.n = 0
        self.child_nodes = None

    def evaluate(self, model: Model) -> float:
        if self.state.is_done():
            value = -1 if self.state.is_lose() else 0
            self.w += value
            self.n += 1
            return value
        if self.child_nodes is None:
            policies, value = predict(model, self.state)
            self.w += value
            self.n += 1
            self.child_nodes = []
            for action, policy in zip(self.state.legal_actions(), policies):
                self.child_nodes.append(Node(self.state.next(action), policy))
            return value
        else:
            value = -self.next_child_node().evaluate(model)

            self.w += value
            self.n += 1
            return value

    def next_child_node(self) -> "Node":
        assert self.child_nodes is not None
        C_PUCT = 1.0
        t = sum(nodes_to_scores(self.child_nodes))
        pucb_values = []
        for child_node in self.child_nodes:
            pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                               C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
        return self.child_nodes[np.argmax(pucb_values)]


def boltzman(xs, temperature):
    xs = [x**(1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


def pv_mcts_scores(model: Model, state: State, temperature: float):
    root_node = Node(state, 0)
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate(model)

    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores


def get_pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state: State):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action


if __name__ == "__main__":
    path = sorted(Path('./model').glob('*.h5'))[-1]
    model = load_model(str(path))
    state = State()
    next_action = get_pv_mcts_action(model, 1.0)

    while not state.is_done():
        action = next_action(state)
        state = state.next(action)
        print(state)
