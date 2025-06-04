import random


class RPSGameEnv:
    def __init__(self):
        self.choices = [0, 1, 2]  # 0 = Pierre, 1 = Feuille, 2 = Ciseaux
        self.states = [(a, b) for a in self.choices for b in self.choices]
        self.start_state = None

    def get_states(self):
        return self.states

    def get_actions(self, state):
        return self.choices

    def is_terminal(self, state):
        return False

    def reset(self):
        self.start_state = random.choice(self.states)
        return self.start_state

    def transition(self, state, action):
        player_first, enemy_first = state
        counter = (player_first + 1) % 3
        enemy_probs = [0.15, 0.15, 0.15]
        enemy_probs[counter] = 0.7
        enemy_action = random.choices(self.choices, weights=enemy_probs)[0]

        # Résultat
        if action == enemy_action:
            reward = 0.0
        elif (action - enemy_action) % 3 == 1:
            reward = 1.0
        else:
            reward = -1.0

        return (action, enemy_action), reward

    def get_transitions(self, state, action):
        next_state, reward = self.transition(state, action)
        return [(1.0, next_state, reward)]

    def get_reward(self, state):
        return 0.0

    def step(self, state, action):
        return self.transition(state, action)
