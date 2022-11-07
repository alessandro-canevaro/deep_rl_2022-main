from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.ex10.mc_evaluate import MCEvaluationAgent
from irlc import PlayWrapper, VideoMonitor, train

if __name__ == "__main__":
    env = BookGridEnvironment(view_mode=1)
    agent = MCEvaluationAgent(env, gamma=.9, alpha=None)
    agent = PlayWrapper(agent, env)
    env.view_mode = 1 # Automatically set value-function view-mode.
    env = VideoMonitor(env, agent=agent, fps=200, render_kwargs={'method_label': 'MC first'})
    train(env, agent, num_episodes=100)
    env.close()
