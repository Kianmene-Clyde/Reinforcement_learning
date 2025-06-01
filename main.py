from environments.line_world import LineWorld
from agents.policy_iteration import policy_iteration

env = LineWorld(length=5)
policy, V = policy_iteration(env)
env.render_policy(policy)  # ou utilise display.py
