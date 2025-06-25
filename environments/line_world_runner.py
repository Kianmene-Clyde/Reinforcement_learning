import pygame
import sys
from Utils.export_results_to_xlsx import export_results
from Utils.save_load_policy import save_policy, load_policy
from agents.dynamic_programming import policy_iteration, value_iteration
from agents.planning_methods import dyna_q, dyna_q_plus
from agents.temporal_difference_methods import sarsa, expected_sarsa, q_learning
from agents.monte_carlo_methods import (
    on_policy_first_visit_mc_control,
    monte_carlo_es,
    off_policy_mc_control
)
from environments.line_world_env import LineWorldEnv

CELL_SIZE = 100
WHITE, BLACK, GREEN, RED, GREY = (255, 255, 255), (0, 0, 0), (50, 200, 50), (200, 50, 50), (180, 180, 180)


class LineWorldRunner:
    def __init__(self, agent_name="Policy Iteration"):
        self.env = LineWorldEnv(length=5)
        self.agent_name = agent_name
        self.policy = {}
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 24)
        self.screen = pygame.display.set_mode((self.env.length * CELL_SIZE, 300))
        pygame.display.set_caption(f"Line World - {self.agent_name}")
        self._prepare_env()

        self.hyperparams_map = {
            "Policy Iteration": {"gamma": 0.99},
            "Value Iteration": {"gamma": 0.99},
            "Dyna Q": {"gamma": 0.95, "alpha": 0.1, "epsilon": 0.1, "planning_steps": 5},
            "Dyna Q+": {"gamma": 0.95, "alpha": 0.1, "epsilon": 0.1, "planning_steps": 5, "kappa": 0.001},
            "Sarsa": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1, "episodes": 100},
            "Expected Sarsa": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1, "episodes": 100},
            "Q Learning": {"gamma": 0.9, "alpha": 0.1, "epsilon": 0.1},
            "First visit Monte Carlo": {"gamma": 0.9, "episodes": 500, "epsilon": 0.1},
            "Monte Carlo ES": {"episodes": 1000},
            "Off-policy Monte Carlo": {"gamma": 0.9, "episodes": 500}
        }

    def _prepare_env(self):
        self.env.get_states = lambda: self.env.states
        self.env.get_actions = lambda s: [0, 1] if s not in self.env.terminal_states else []
        self.env.is_terminal = lambda s: s in self.env.terminal_states
        self.env.get_transitions = lambda s, a: [(1.0, *self.env.transition(s, a))]
        self.env.num_states = self.env.length
        self.env.num_actions = 2

    def run(self):
        print(f"Bienvenue dans l'environnement Line World avec l'agent '{self.agent_name}'")
        print("Souhaitez-vous :")
        print("1 - Charger une politique existante")
        print("2 - Apprendre une nouvelle politique")
        choix = input("Votre choix (1/2) : ").strip()

        if choix == "1":
            try:
                filename = f"lineworld_{self.agent_name.replace(' ', '_').lower()}.pkl"
                self.policy = load_policy(filename)
            except FileNotFoundError as e:
                print(e)
                return
        else:
            agent_func_map = {
                "Policy Iteration": policy_iteration,
                "Value Iteration": value_iteration,
                "Dyna Q": dyna_q,
                "Dyna Q+": dyna_q_plus,
                "Sarsa": sarsa,
                "Expected Sarsa": expected_sarsa,
                "First visit Monte Carlo": on_policy_first_visit_mc_control,
                "Monte Carlo ES": monte_carlo_es,
                "Off-policy Monte Carlo": off_policy_mc_control,
                "Q Learning": q_learning,
            }

            if self.agent_name not in agent_func_map:
                print(f"⚠️ Agent '{self.agent_name}' non reconnu.")
                return

            agent_func = agent_func_map[self.agent_name]
            hyperparams = self.hyperparams_map.get(self.agent_name, {})

            try:
                self.policy, _ = agent_func(self.env, **hyperparams)
            except Exception as e:
                print(f"Erreur lors de l'exécution de l'agent {self.agent_name} : {e}")
                return

            print("Politique entraînée. Souhaitez-vous la sauvegarder ? (O/N)")
            if input().strip().lower() == "o":
                filename = f"lineworld_{self.agent_name.replace(' ', '_').lower()}.pkl"
                save_policy(self.policy, filename)

        print("Choisissez un mode :")
        print("1 - Agent automatique")
        print("2 - Joueur humain")
        mode = input("Mode : ").strip()

        if mode == "1":
            self._run_agent()
        elif mode == "2":
            self._run_human()
        else:
            print("Mode invalide.")

    def _draw_arrow(self, surface, x, y, direction):
        if direction == 1:
            pygame.draw.polygon(surface, BLACK, [(x, y), (x - 10, y - 10), (x - 10, y + 10)])
        elif direction == 0:
            pygame.draw.polygon(surface, BLACK, [(x, y), (x + 10, y - 10), (x + 10, y + 10)])

    def _draw(self, policy, score, episode, message=""):
        self.screen.fill(WHITE)
        for i in self.env.states:
            rect = pygame.Rect(i * CELL_SIZE + 10, 100, CELL_SIZE - 20, 100)
            color = RED if i == 0 else GREEN if i == self.env.length - 1 else GREY
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, BLACK, rect, 2)
            label = self.font.render(str(i), True, BLACK)
            self.screen.blit(label, (i * CELL_SIZE + CELL_SIZE // 2 - 5, 110))
            if i not in self.env.terminal_states and i in policy:
                self._draw_arrow(self.screen, i * CELL_SIZE + CELL_SIZE // 2, 210, policy[i])

        x = self.env.agent_pos * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, (0, 0, 255), (x, 170), 20)

        self.screen.blit(self.font.render(f"Score: {score:.1f}", True, BLACK), (10, 20))
        self.screen.blit(self.font.render(f"Épisode: {episode}", True, BLACK), (200, 20))
        if message:
            self.screen.blit(self.font.render(message, True, BLACK), (10, 60))
        pygame.display.flip()

    def _wait_for_restart(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

    def _run_agent(self):
        episode = 1
        while True:
            state = self.env.reset()
            total_reward = 0
            while True:
                self._draw(self.policy, total_reward, episode)
                pygame.time.delay(400)
                if state in self.env.terminal_states:
                    self._draw(self.policy, total_reward, episode, "✅ Terminé - appuyez sur R")
                    break
                action = self.policy[state]
                next_state, reward = self.env.transition(state, action)
                state = next_state
                total_reward += reward
                self.env.agent_pos = state
            self._draw(self.policy, total_reward, episode, "✅ Terminé - appuyez sur R")

            # ✅ EXPORT des résultats
            export_results(
                agent_name=self.agent_name,
                env_name="LineWorld",
                stats={"score_total": total_reward, "nb_episodes": episode},
                hyperparams=self.hyperparams_map.get(self.agent_name, {})
            )

            episode += 1
            self._wait_for_restart()

    def _run_human(self):
        episode = 1
        while True:
            state = self.env.reset()
            total_reward = 0
            while True:
                self._draw({}, total_reward, episode)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if state in self.env.terminal_states:
                            break
                        if event.key == pygame.K_LEFT:
                            next_state, reward = self.env.transition(state, 0)
                        elif event.key == pygame.K_RIGHT:
                            next_state, reward = self.env.transition(state, 1)
                        else:
                            continue
                        state = next_state
                        total_reward += reward
                        self.env.agent_pos = state
                if state in self.env.terminal_states:
                    break
            self._draw({}, total_reward, episode, "✅ Terminé - appuyez sur R")
            episode += 1
            self._wait_for_restart()
