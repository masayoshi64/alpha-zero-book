import random
import math


class State:
    def __init__(self, pieces=None, enemy_pieces=None) -> None:
        self.pieces = pieces if pieces is not None else [0] * 9
        self.enemy_pieces = enemy_pieces \
            if enemy_pieces is not None else [0] * 9

    def pieces_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    def is_lose(self):
        def is_comp(x, y, dx, dy):
            for k in range(3):
                if y < 0 or 2 < y or x < 0 or 2 < x or self.enemy_pieces[x + y * 3] == 0:
                    return False
                x, y = x + dx, y + dy
            return True
        if is_comp(0, 0, 1, 1) or is_comp(0, 2, 1, -1):
            return True
        for i in range(3):
            if is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1):
                return True
        return False

    def is_draw(self):
        return self.pieces_count(self.pieces) + \
            self.pieces_count(self.enemy_pieces) == 9

    def is_done(self):
        return self.is_lose() or self.is_draw()

    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    def is_first_player(self):
        return self.pieces_count(
            self.pieces) == self.pieces_count(
            self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(9):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += ' '
            if(i % 3 == 2):
                str += '\n'
        return str


def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


def argmax(collection, key=None):
    return collection.index(max(collection))


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


def playout(state: State):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    return -playout(state.next(random_action(state)))


class Node:
    def __init__(self, state: State) -> None:
        self.state = state
        self.w = 0
        self.n = 0
        self.child_nodes = None

    def evaluate(self):
        if self.state.is_lose():
            value = -1 if self.state.is_lose() else 0
            self.w += value
            self.n += 1
            return value

        if not self.child_nodes:
            value = playout(self.state)
            self.w += value
            self.n += 1

            if self.n == 10:
                self.expand()
            return value
        else:
            value = -self.next_child_node().evaluate()

            self.w += value
            self.n += 1
            return value

    def expand(self):
        legal_actions = self.state.legal_actions()
        self.child_nodes = []
        for action in legal_actions:
            self.child_nodes.append(Node(self.state.next(action)))

    def next_child_node(self):
        assert(self.child_nodes is not None)
        for child_node in self.child_nodes:
            if child_node.n == 0:
                return child_node
        t = 0
        for c in self.child_nodes:
            t += c.n
        ucb1_values = []
        for child_node in self.child_nodes:
            ucb1_values.append(-child_node.w / child_node.n +
                               (2 * math.log(t) / child_node.n)**0.5)
        return self.child_nodes[argmax(ucb1_values)]


def mcts_action(state):
    root_node = Node(state)
    root_node.expand()

    for _ in range(100):
        root_node.evaluate()

    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:  # type: ignore
        n_list.append(c.n)

    return legal_actions[argmax(n_list)]
