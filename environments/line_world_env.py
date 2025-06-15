class LineWorldEnv:
    def __init__(self, length=5):
        self.length = length
        self.states = list(range(length))
        self.terminal_states = [0, length - 1]
        self.agent_pos = 2

        # ✅ Interface standard pour agents tabulaires
        self.num_states = length
        self.num_actions = 2
        self._score = 0  # utile pour TD

    def reset(self, pos=2):
        self.agent_pos = pos
        self._score = 0
        return self.agent_pos

    def reset_to(self, state_index, action):
        self.agent_pos = state_index
        self._score = 0
        return self.agent_pos

    def step(self, action):
        next_state, reward = self.transition(self.agent_pos, action)
        self._score += reward
        self.agent_pos = next_state
        return next_state, reward

    def get_state(self):
        return self.agent_pos

    def score(self):
        return self._score

    def is_game_over(self):
        return self.agent_pos in self.terminal_states

    def get_states(self):
        return self.states

    def get_actions(self, state):
        return [0, 1] if state not in self.terminal_states else []

    def is_terminal(self, state):
        return state in self.terminal_states

    def transition(self, state, action):
        if state in self.terminal_states:
            return state, 0.0
        next_state = max(0, state - 1) if action == 0 else min(self.length - 1, state + 1)
        reward = -1.0 if next_state == 0 else 1.0 if next_state == self.length - 1 else 0.0
        return next_state, reward

    def get_transitions(self, state, action):
        next_state, reward = self.transition(state, action)
        return [(1.0, next_state, reward)]
