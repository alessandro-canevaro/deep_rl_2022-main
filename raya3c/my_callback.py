

"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict, Tuple
import argparse
import numpy as np
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=2000)


class MyCallbacks(DefaultCallbacks):
    wandb = None
    def on_algorithm_init(self, *args, **kwargs):
        print("Initializing the callback logger..")
        if self.wandb is None:
            import wandb
            wandb.init(project="VPN", entity="deep-rl", name="run1")
            wandb.config = {
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 128
            }
            # for loss in range(10):
            #     wandb.log({"loss": np.sqrt(loss)})
            # wandb.watch(model, log_freq=100)
            self.wandb = wandb

        a = 234
        pass


    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.user_data["pole_angles"] = []
        episode.hist_data["pole_angles"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        pole_angle = abs(episode.last_observation_for()[2])
        raw_angle = abs(episode.last_raw_obs_for()[2])
        assert pole_angle == raw_angle
        episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        pole_angle = np.mean(episode.user_data["pole_angles"])
        # print(
        #     "episode {} (env-idx={}) ended with length {} and pole "
        #     "angles {}".format(
        #         episode.episode_id, env_index, episode.length, pole_angle
        #     )
        # )
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        pass
        # print("on_sample_end, returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # print("Algorithm.train() result: {} -> {} episodes".format(algorithm, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True
        lstats = result['info']['learner']
        if 'default_policy' in lstats:
            lstats = lstats['default_policy']['learner_stats']
            # print("> CALLBACK HAD NON-ZERO INFO", lstats)

        else:
            # print("no l stats")
            lstats = {}

        hstats = {k: np.mean(v) for k, v in result['hist_stats'].items()}

        stats = lstats | hstats |  result['timers'] | result['counters']
        # print(stats)
        # print(hstats)
        self.wandb.log( stats)

        # print("ON TRAIN", result['info'])


    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
        print(
            "policy.learn_on_batch() result: {} -> sum actions: {}".format(
                policy, result["sum_actions_in_train_batch"]
            )
        )
        print("learn on batch", result['info'] )


        lstats = result['info']['learner']
        if 'default_policy' in lstats:
            lstats = lstats['default_policy']['learner_stats']
        else:
            # print("no l stats")
            lstats = {}





        # print("INFO IS", result['info'])

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
