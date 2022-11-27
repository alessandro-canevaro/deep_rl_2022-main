from irlc import PlayWrapper, VideoMonitor, train


def keyboard_play_value(env, agent, method_label='MC',  q=False):
    agent = PlayWrapper(agent, env)
    env.view_mode = 1 # Set value-function view-mode.
    env = VideoMonitor(env, agent=agent, fps=200, continious_recording=True, render_kwargs={'method_label': method_label})
    env.view_mode = 1
    train(env, agent, num_episodes=100)
    env.close()