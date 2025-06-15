class GridWorldEnv:
    def __init__(self, width=5, height=5, start=(0, 0), walls=None):
        self.width = width
        self.height = height
        self.start = start
        self.terminal_states = [(0, 4), (4, 4)]
        self.walls = walls if walls is not None else []
        self.rewards = {
            (0, 4): -3.0,
            (4, 4): 1.0,
        }

        self.agent_pos = start
        self.actions = {
            0: (-1, 0),  # Haut
            1: (1, 0),  # Bas
            2: (0, -1),  # Gauche
            3: (0, 1),  # Droite
        }

        self.num_actions = len(self.actions)
        self.states = self.get_states()
        self.state_to_index = {s: i for i, s in enumerate(self.states)}
        self.index_to_state = {i: s for s, i in self.state_to_index.items()}
        self.num_states = len(self.states)
        self._score = 0

    def reset(self, pos=None):
        self.agent_pos = pos if pos is not None else self.start
        self._score = 0
        return self.agent_pos

    def reset_to(self, state_index, action):
        self.agent_pos = self.index_to_state[state_index]
        self._score = 0
        return self.agent_pos

    def step(self, action):
        next_state, reward = self.transition(self.agent_pos, action)
        self._score += reward
        self.agent_pos = next_state
        return next_state, reward

    def get_state(self):
        return self.state_to_index[self.agent_pos]

    def score(self):
        return self._score

    def is_game_over(self):
        return self.agent_pos in self.terminal_states

    def get_states(self):
        return [(x, y) for x in range(self.height) for y in range(self.width) if (x, y) not in self.walls]

    def get_actions(self, state):
        return [] if self.is_terminal(state) else list(self.actions.keys())

    def is_terminal(self, state):
        return state in self.terminal_states

    def transition(self, state, action):
        if self.is_terminal(state):
            return state, 0.0

        dx, dy = self.actions[action]
        next_state = (state[0] + dx, state[1] + dy)

        if (
                0 <= next_state[0] < self.height and
                0 <= next_state[1] < self.width and
                next_state not in self.walls
        ):
            pass
        else:
            next_state = state

        reward = self.rewards.get(next_state, 0.0)
        return next_state, reward

    def get_transitions(self, state, action):
        next_state, reward = self.transition(state, action)
        return [(1.0, next_state, reward)]

    def get_reward(self, state):
        return self.rewards.get(state, 0.0)
