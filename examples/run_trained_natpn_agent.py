"""
The main script for evaluating a natpn policy in an environment.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from robomimic.utils.dataset import SequenceDataset
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.algo.bc import BC_NatPN

import cv2
import argparse
import h5py
import imageio
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase
from tqdm import tqdm

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], uncertainties=[], log_probs=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            pred = policy(ob=obs)
            act = pred["actions"]
            uncertainty = pred["uncertainty"]
            log_prob = pred["log_prob"]
            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        # video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))

                        frame = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        text1 = f"Uncertainty: {uncertainty:.2f}"
                        text2 = f"LogProb: {log_prob:.2f}"
                        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        video_img.append(frame)

                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            traj["uncertainties"].append(uncertainty)
            traj["log_probs"].append(log_prob)
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj
    
def run_rollout(policy, ckpt_dict, num_rollout_episodes=10,rollout_horizon=400,video_path=None):
    
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=None, 
        render=True, 
        render_offscreen=False, 
        verbose=True,
    )
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    rollout_stats = []
    trajectories = []
    for i in tqdm(range(num_rollout_episodes), desc="Rollouts"):
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=False, 
            video_writer=video_writer, 
            video_skip=1, 
            return_obs=False,
            camera_names=["agentview"],
        )
        rollout_stats.append(stats)
        trajectories.append(traj)

        # # visualize the uncertainty and log_prob separately
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(traj["uncertainties"])
        # plt.title("Uncertainty")
        # plt.xlabel("Step")
        # plt.ylabel("Uncertainty")
        # plt.subplot(1, 2, 2)
        # plt.plot(traj["log_probs"])
        # plt.title("Log Probability")
        # plt.xlabel("Step")
        # plt.ylabel("Log Probability")
        # plt.tight_layout()
    if video_path is not None:
        video_writer.close()
        

    return rollout_stats


if __name__ == "__main__":

    config_path = "/home/han/projects/robomimic/bc_trained_models/test/20250520093418/config.json"
    video_dir = "/home/han/projects/robomimic/bc_trained_models/videos"
    ckpt_path = "/home/han/projects/robomimic/bc_trained_models/test/20250520093418/models/model_epoch_450.pth"

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
  
    all_rollout_logs = run_rollout(
        policy=policy,
        ckpt_dict = ckpt_dict,
        num_rollout_episodes=10,
        rollout_horizon=600,
        video_path=os.path.join(video_dir, "natpn_toolhang_radial_450.mp4"),
    )
    success_rate = 0
    for i, rollout_log in enumerate(all_rollout_logs):
        success_rate += rollout_log["Success_Rate"]
    success_rate /= len(all_rollout_logs)
    print(f"model: {ckpt_path}")
    print(f"Success Rate: {success_rate}")
    print("Rollout complete.")