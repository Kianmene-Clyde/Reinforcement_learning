import random


class MontyHallEnv:
    def __init__(self):
        self.doors = [0, 1, 2]
        self.states = []  # État = (étape, porte_choisie, porte_ouverte)
        self._build_states()
        self.reset()

    def _build_states(self):
        # start: choix initial
        for first_choice in self.doors:
            self.states.append(("start", first_choice, None))

        # reveal: une porte est ouverte, deux choix: garder ou changer
        for first_choice in self.doors:
            for opened in self.doors:
                if opened != first_choice:
                    self.states.append(("reveal", first_choice, opened))

        # done: état terminal, succès ou échec
        for final_choice in self.doors:
            self.states.append(("done", final_choice))

    def get_states(self):
        return self.states

    def get_actions(self, state):
        phase = state[0]
        if phase == "start":
            return self.doors  # choisir une porte
        elif phase == "reveal":
            return [0, 1]  # 0=garder, 1=changer
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
            possible_doors = [d for d in self.doors if d != chosen_door and d != self.winning_door]
            revealed_door = random.choice(possible_doors)
            next_state = ("reveal", chosen_door, revealed_door)
            return next_state, 0.0

        elif phase == "reveal":
            chosen_door = state[1]
            revealed_door = state[2]

            if action == 0:  # garder
                final_choice = chosen_door
            else:  # changer
                final_choice = [d for d in self.doors if d not in [chosen_door, revealed_door]][0]

            reward = 1.0 if final_choice == self.winning_door else 0.0
            next_state = ("done", final_choice)
            return next_state, reward

        else:
            return state, 0.0

    def get_transitions(self, state, action):
        # Pour policy iteration — retourne une liste de tuples (proba, next_state, reward)
        next_state, reward = self.step(state, action)
        return [(1.0, next_state, reward)]

    def get_reward(self, state):
        if state[0] == "done":
            return 1.0 if state[1] == self.winning_door else 0.0
        return 0.0
