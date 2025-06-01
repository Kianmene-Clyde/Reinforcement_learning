from environments.line_world import LineWorld
from agents.policy_iteration import policy_iteration

def test_line_world():
    env = LineWorld(length=5)
    policy, V = policy_iteration(env)
    env.render_policy(policy)

if __name__ == "__main__":
    test_line_world()
