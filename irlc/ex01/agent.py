# from line_profiler_pycharm import profile
import inspect
import itertools
import os
import sys
from collections import OrderedDict, namedtuple
import numpy as np
from tqdm import tqdm
from irlc.utils.common import load_time_series, log_time_series
from irlc.utils.irlc_plot import existing_runs
import shutil
from time import sleep
# Main agent class. See for further details. for additional details.
from gym import Env

class Agent: #!s #!s
    """
    Main agent class. See \nref{c1s43} for additional details.

    :Example:

    >>> print("Hello World")
    "Hello world"

HERE WE GO WITH SOME OF THE COOL MATH.
.. math::
    :nowrap:

    \\begin{eqnarray}
    y    & = & ax^2 + bx + c \\\\
    f(x) & = & x^2 + 2xy + y^2
    \\end{eqnarray}
    """
    
    def __init__(self, env: Env):#!s
        """
        Instantiate the Agent class.

        Args:
            env: An openai gym Environment instance.
        """
        self.env = env   #!s

    def pi(self, s, k=None):
        """ Evaluate the Agent's policy at time step `k` in state `s`
        
        The details will differ depending on whether the agent interacts in a discrete-time or continous-time setting. 
        
        - For discrete application (dynamical programming/search and reinforcement learning), k is discrete k=0, 1, 2, ...
        - For control applications, k is continious and denote simulation time t, i.e. it should be called as
        > agent.pi(x, t) #!s

        :param s: Current state
        :param k: Current time index.
        :return: action
        """ #!s
        return self.env.action_space.sample() #!s

    def train(self, s, a, r, sp, done=False): #!s
        """
        Called at each step of the simulation after a = pi(s,k) and environment transition to sp. 
        
        Allows the agent to learn from experience  #!s

        :param s: Current state x_k
        :param a: Action taken
        :param r: Reward obtained by taking action a_k in x_k
        :param sp: The state that the environment transitioned to :math:`{\\bf x}_{k+1}`
        :param done: Whether environment terminated when transitioning to sp
        :return: None
        """ #!s
        pass #!s

    def __str__(self):
        """ Optional: A unique name for this agent. Used for labels when plotting, but can be kept like this. """
        return super().__str__()

    def extra_stats(self):
        """ Optional: Can be used to record extra information from the Agent while training.
        You can safely ignore this method, it will only be used for control theory to create nicer plots """
        return {}

    def hello(self, greeting: str) -> str:
        """ The canonical hello world example.

        A *longer* description with some **RST**.

        Args:
            greeting: The person to say hello to.
        Returns:
            str: The greeting
         """
        return f"{greeting} {self.name}"

fields = ('time', 'state', 'action', 'reward')
Trajectory = namedtuple('Trajectory', fields + ("env_info",))


def train(env, agent=None, experiment_name=None, num_episodes=1, verbose=True,
          reset=True, # If True we will call env.reset() upon episode start.
          max_steps=1e10,
          max_runs=None,
          return_trajectory=True, # Return the current trajectories as a list
          resume_stats=None, # Resume stat collection from last save.
          log_interval=1, # Only log every log_interval steps. Reduces size of log files.
          delete_old_experiments=False, # Remove the old experiments folder. Useful while debugging a model (or to conserve disk space)
          sleep_time=0,
          ):
    """
    Implement the main training loop, see \nref{c1s44}.
    Simulate the interaction between agent `agent` and the environment `env`.
    The function has a lot of special functionality, so it is useful to consider the common cases. An example:

    >>> stats, _ = train(env, agent, num_episodes=2)

    Simulate interaction for two episodes (i.e. environment terminates two times and is reset).
    `stats` will be a list of length two containing information from each run

    >>> stats, trajectories = train(env, agent, num_episodes=2, return_Trajectory=True)

    `trajectories` will be a list of length two containing information from the two trajectories.

    >>> stats, _ = train(env, agent, experiment_name='experiments/my_run', num_episodes=2)

    Save `stats`, and trajectories, to a file which can easily be loaded/plotted (see course software for examples of this).
    The file will be time-stamped so using several calls you can repeat the same experiment (run) many times.

    >>> stats, _ = train(env, agent, experiment_name='experiments/my_run', num_episodes=2, max_runs=10)

    As above, but do not perform more than 10 runs. Useful for repeated experiments.

    :param env: Environment (Gym instance)
    :param agent: Agent instance
    :param experiment_name: Save outcome to file for easy plotting (Optional)
    :param num_episodes: Number of episodes to simulate
    :param verbose: Display progress bar
    :param reset: Call env.reset() before simulation start.
    :param max_steps: Terminate if this many steps have elapsed (for non-terminating environments)
    :param max_runs: Maximum number of repeated experiments (requires `experiment_name`)
    :param return_trajectory: Return trajectories list (Off by default since it might consume lots of memory)
    :param resume_stats: Resume stat collection from last run (requires `experiment_name`)
    :param log_interval: Log stats less frequently
    :param delete_old_experiment: Delete old experiment with the same name
    :return: stats, trajectories (both as lists)
    """

    from irlc import cache_write
    from irlc import cache_read
    saveload_model = False
    temporal_policy = None
    save_stats = True
    if agent is None:
        print("[train] No agent was specified. Using irlc.Agent(env) (this agent selects actions at random)")
        agent = Agent(env)

    if delete_old_experiments and experiment_name is not None and os.path.isdir(experiment_name):
        shutil.rmtree(experiment_name)

    if experiment_name is not None and max_runs is not None and existing_runs(experiment_name) >= max_runs:
        stats, recent = load_time_series(experiment_name=experiment_name)
        if return_trajectory:
            trajectories = cache_read(recent+"/trajectories.pkl")
        else:
            trajectories = []
        return stats, trajectories
    stats = []
    steps = 0
    ep_start = 0
    resume_stats = saveload_model if resume_stats is None else resume_stats

    recent = None
    if resume_stats:
        stats, recent = load_time_series(experiment_name=experiment_name)
        if recent is not None:
            ep_start, steps = stats[-1]['Episode']+1, stats[-1]['Steps']
    if temporal_policy is None:
        a = inspect.getfullargspec(agent.pi)
        temporal_policy = len(a.args) >= 3  # does the policy include a time step?

    trajectories = []
    include_metadata = len(inspect.getfullargspec(agent.train).args) >= 7
    break_outer = False

    with tqdm(total=num_episodes, disable=not verbose, file=sys.stdout) as tq:
        for i_episode in range(num_episodes): #!s=train
            if break_outer:
                break
            if reset or i_episode > 0:
                s = env.reset() #!s=train
            elif hasattr(env, "s"):
                s = env.s
            elif hasattr(env, 'state'):
                s = env.state
            else:
                s = env.model.s
            time = 0 #!s=train
            reward = []
            trajectory = Trajectory(time=[], state=[], action=[], reward=[], env_info=[])

            for _ in itertools.count():
                try:
                    #agent.plotvalues()
                    sleep(sleep_time) #for visualization
                except AttributeError: #the agent doesnt have the plotvalue attribute
                    pass
                a = agent.pi(s,time) if temporal_policy else agent.pi(s)
                sp, r, done, metadata = env.step(a)

                if not include_metadata:
                    agent.train(s, a, r, sp, done)
                else:
                    agent.train(s, a, r, sp, done, metadata)

                if return_trajectory:
                    trajectory.time.append(np.asarray(time))
                    trajectory.state.append(s)
                    trajectory.action.append(a)
                    trajectory.reward.append(np.asarray(r))
                    trajectory.env_info.append(metadata)

                reward.append(r)
                steps += 1

                time += metadata['dt'] if 'dt' in metadata else 1
                if done or steps >= max_steps:
                    trajectory.state.append(sp)
                    trajectory.time.append(np.asarray(time))
                    break_outer = steps >= max_steps
                    break
                s = sp #!s=train
            if return_trajectory:
                try:
                    trajectory = Trajectory(**{field: np.stack([np.asarray(x_) for x_ in getattr(trajectory, field)]) for field in fields}, env_info=trajectory.env_info)
                except Exception as e:
                    pass

                trajectories.append(trajectory)
            if (i_episode + 1) % log_interval == 0:
                stats.append({"Episode": i_episode + ep_start,
                              "Accumulated Reward": sum(reward),
                              "Average Reward": np.mean(reward),
                              "Length": len(reward),
                              "Steps": steps, # Steps is deprecated; pending removal.
                              **agent.extra_stats()})

            tq.set_postfix(ordered_dict=OrderedDict(list(OrderedDict(stats[-1]).items() )[:5] )) if len(stats) > 0 else None
            tq.update()
    sys.stderr.flush()

    if resume_stats and save_stats and recent is not None:
        os.remove(recent+"/log.txt")

    if experiment_name is not None and save_stats:
        path = log_time_series(experiment=experiment_name, list_obs=stats)
        if return_trajectory:
            cache_write(trajectories, path+"/trajectories.pkl")

        print(f"Training completed. Logging {experiment_name}: '{', '.join( stats[0].keys()) }'")

    for i, t in enumerate(trajectories):
        from collections import defaultdict
        nt = defaultdict(lambda: [])
        if "supersample" in t.env_info[0]:
            for f in fields:
                for k, ei in enumerate(t.env_info):
                    z = ei['supersample'].__getattribute__(f).T
                    if k == 0:
                        pass
                    else:
                        z = z[1:]
                    nt[f].append(z)

            for f in fields:
                nt[f] = np.concatenate([z for z in nt[f]],axis=0)
            traj2 = Trajectory(**nt, env_info=[])
            trajectories[i] = traj2

    # Turn this into a single episodes-list (refactor later)


    return stats, trajectories
