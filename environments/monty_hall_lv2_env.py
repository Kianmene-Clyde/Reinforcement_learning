import random


class MontyHallEnvLv2:
    def __init__(self):
        self.doors = list(range(5))
        self.states = []
        self._build_states()
        self.state_to_index = {s: i for i, s in enumerate(self.states)}
        self.index_to_state = {i: s for s, i in self.state_to_index.items()}
        self.num_states = len(self.states)
        self.num_actions = 5  # nombre max d'actions possibles (choisir une porte)

        self.reset()
        self._score = 0

    def _build_states(self):
        for first_choice in self.doors:
            self.states.append(("start", first_choice, None))

        for first_choice in self.doors:
            for revealed_combo in self._combinations_except(first_choice):
                self.states.append(("reveal", first_choice, tuple(sorted(revealed_combo))))

        for final_choice in self.doors:
            self.states.append(("done", final_choice))

    def _combinations_except(self, chosen):
        combis = []
        for w in self.doors:
            if w == chosen:
                continue
            remaining = [d for d in self.doors if d != chosen and d != w]
            if len(remaining) >= 3:
                combis.append(random.sample(remaining, 3))
        return combis

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.agent_state = ("start", random.choice(self.doors), None)
        self._score = 0
        return self.agent_state

    def reset_to(self, state_index, action):
        self.winning_door = random.choice(self.doors)
        self.agent_state = self.index_to_state[state_index]
        self._score = 0
        return self.agent_state

    def get_states(self):
        return self.states

    def get_actions(self, state):
        phase = state[0]
        if phase == "start":
            return self.doors
        elif phase == "reveal":
            return [d for d in self.doors if d not in state[2]]
        else:
            return []

    def is_terminal(self, state):
        return state[0] == "done"

    def transition(self, state, action):
        phase = state[0]

        if phase == "start":
            chosen_door = action
            possible = [d for d in self.doors if d != chosen_door and d != self.winning_door]
            revealed = tuple(sorted(random.sample(possible, 3)))
            return ("reveal", chosen_door, revealed), 0.0

        elif phase == "reveal":
            final_choice = action
            reward = 1.0 if final_choice == self.winning_door else 0.0
            return ("done", final_choice), reward

        else:
            return state, 0.0

    def get_transitions(self, state, action):
        next_state, reward = self.transition(state, action)
        return [(1.0, next_state, reward)]

    def get_reward(self, state):
        if state[0] == "done":
            return 1.0 if state[1] == self.winning_door else 0.0
        return 0.0

    # === Compatibilité agents tabulaires ===
    def step(self, state, action):
        next_state, reward = self.transition(state, action)
        self.agent_state = next_state
        return next_state, reward

    def get_state(self):
        return self.state_to_index[self.agent_state]

    def score(self):
        return self._score

    def is_game_over(self):
        return self.agent_state[0] == "done"
