import random
from itertools import product


class RPSGameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = None
        self.round = 1
        self.done = False
        return self.state

    def get_actions(self, state):
        if state == "TERMINAL":
            return []
        return [0, 1, 2]  # Pierre, Feuille, Ciseaux

    def is_terminal(self, state):
        return state == "TERMINAL"

    def transition(self, state, action):
        if self.done:
            return "TERMINAL", 0.0

        if self.round == 1:
            enemy_action = 0  # déterministe pour les algos DP
            next_state = (action, enemy_action)
            reward = self.get_reward(action, enemy_action)
            self.round = 2
            self.state = next_state
            return next_state, reward
        else:
            counter = self.counter_action(state[0])
            enemy_probs = [0.15, 0.15, 0.15]
            enemy_probs[counter] = 0.7
            enemy_action = counter  # idem, déterministe pour DP

            reward = self.get_reward(action, enemy_action)
            self.state = "TERMINAL"
            self.done = True
            return "TERMINAL", reward

    def get_reward(self, player, enemy):
        if player == enemy:
            return 0.0
        elif (player == 0 and enemy == 2) or (player == 1 and enemy == 0) or (player == 2 and enemy == 1):
            return 1.0
        else:
            return -1.0

    def counter_action(self, action):
        return (action + 1) % 3

    def get_states(self):
        return [None] + list(product([0, 1, 2], repeat=2)) + ["TERMINAL"]

    def get_transitions(self, state, action):
        if state == "TERMINAL":
            return []

        if state is None:
            enemy_action = 0  # pour simplifier, comme si toujours même ennemi
            next_state = (action, enemy_action)
            reward = self.get_reward(action, enemy_action)
            return [(1.0, next_state, reward)]

        # round 2
        player_first = state[0]
        counter = self.counter_action(player_first)
        enemy_probs = [0.15, 0.15, 0.15]
        enemy_probs[counter] = 0.7
        transitions = []
        for enemy_action, prob in enumerate(enemy_probs):
            reward = self.get_reward(action, enemy_action)
            transitions.append((prob, "TERMINAL", reward))
        return transitions
