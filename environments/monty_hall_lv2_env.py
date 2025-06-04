import random


class MontyHallEnvLv2:
    def __init__(self):
        self.doors = list(range(5))
        self.states = []
        self._build_states()
        self.reset()

    def _build_states(self):
        # start phase
        for first_choice in self.doors:
            self.states.append(("start", first_choice, None))

        # reveal phase
        for first_choice in self.doors:
            for revealed_combo in self._combinations_except(first_choice):
                self.states.append(("reveal", first_choice, tuple(sorted(revealed_combo))))

        # done phase
        for final_choice in self.doors:
            self.states.append(("done", final_choice))

    def _combinations_except(self, chosen):
        # Possible 3-door reveals, excluding chosen and winning
        combis = []
        for w in self.doors:
            if w == chosen:
                continue
            remaining = [d for d in self.doors if d != chosen and d != w]
            if len(remaining) >= 3:
                combis.append(random.sample(remaining, 3))
        return combis

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

    def reset(self):
        self.winning_door = random.choice(self.doors)
        first_choice = random.choice(self.doors)
        return ("start", first_choice, None)

    def step(self, state, action):
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
        next_state, reward = self.step(state, action)
        return [(1.0, next_state, reward)]

    def get_reward(self, state):
        if state[0] == "done":
            return 1.0 if state[1] == self.winning_door else 0.0
        return 0.0
