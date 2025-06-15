from itertools import product


class RPSGameEnv:
    def __init__(self):
        self.actions = [0, 1, 2]  # 0: Pierre, 1: Feuille, 2: Ciseaux
        self.states = [None] + list(product(self.actions, repeat=2)) + ["TERMINAL"]
        self.state_to_index = {s: i for i, s in enumerate(self.states)}
        self.index_to_state = {i: s for s, i in self.state_to_index.items()}
        self._score = 0
        self.reset()

    def reset(self):
        self.state = None
        self.round = 1
        self.done = False
        self._score = 0
        return self.state

    def get_states(self):
        return self.states

    def get_actions(self, state):
        if state == "TERMINAL":
            return []
        return [0, 1, 2]

    def is_terminal(self, state):
        return state == "TERMINAL"

    def transition(self, state, action):
        if self.done:
            return "TERMINAL", 0.0

        if self.round == 1:
            enemy_action = 0  # fixe pour simplicité
            next_state = (action, enemy_action)
            reward = self.get_reward(action, enemy_action)
            self.state = next_state
            self.round = 2
            return next_state, reward

        else:
            counter = self.counter_action(state[0])
            enemy_action = counter  # fixe aussi
            reward = self.get_reward(action, enemy_action)
            self.state = "TERMINAL"
            self.done = True
            return "TERMINAL", reward

    def get_transitions(self, state, action):
        if state == "TERMINAL":
            return []

        if state is None:
            enemy_action = 0
            next_state = (action, enemy_action)
            reward = self.get_reward(action, enemy_action)
            return [(1.0, next_state, reward)]
        else:
            counter = self.counter_action(state[0])
            enemy_action = counter
            reward = self.get_reward(action, enemy_action)
            return [(1.0, "TERMINAL", reward)]

    def get_reward(self, player, enemy):
        if player == enemy:
            return 0.0
        elif (player == 0 and enemy == 2) or (player == 1 and enemy == 0) or (player == 2 and enemy == 1):
            return 1.0
        else:
            return -1.0

    def counter_action(self, action):
        return (action + 1) % 3

    # === Interface pour les agents tabulaires ===
    def step(self, action):
        next_state, reward = self.transition(self.state, action)
        self.state = next_state
        self._score += reward

    def get_state(self):
        return self.state_to_index[self.state]

    def score(self):
        return self._score

    def is_game_over(self):
        return self.state == "TERMINAL"

    def num_states(self):
        return len(self.states)

    def num_actions(self):
        return len(self.actions)
