import random


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


def evaluate_algorithm_of(label, next_actions, game_count):
    total_point = 0
    for i in range(game_count):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        print(f"Evaluate {i+1}/{game_count}\r", end='')
    print('')

    average_point = total_point / game_count
    print(label.format(average_point))
