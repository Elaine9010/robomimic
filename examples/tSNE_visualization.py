"""
The main script for evaluating a natpn policy in an environment.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import OrderedDict
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils


from copy import deepcopy


def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.
    """
    # Create training dataset and loader
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions", 
            "rewards", 
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,                  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute="train",       # can optionally provide a filter key here
    )
    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )

    return data_loader


def run_encoder(model, data_loader):

    flow = model.nets["policy"].flow
    latent_vectors = [] #list to store latent vectors
    log_probs = [] #list to store log probabilities
    
    for batch in data_loader:
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)
        obs = model.nets["encoder"](
            obs=input_batch["obs"],
            goal = input_batch.get("goal_obs", None),
        )
        with torch.no_grad():  # freeze encoder
            z = obs.detach().float()

        latent_vectors.append(z.cpu().numpy())

        log_prob = flow(z)
        log_prob = log_prob.detach().float()
        log_probs.append(log_prob.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    log_probs = np.concatenate(log_probs, axis=0)
    
    return latent_vectors, log_probs

def visualize_latent_space(latent_vectors, log_probs,save_path=None):
    # Apply t-SNE to the latent vectors
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(latent_vectors)

    # Normalize the log_prob values to use them as colors
    # log_prob_min = np.min(log_probs)
    # log_prob_max = np.max(log_probs)
    
    # log_probs_normalized = (log_probs - log_prob_min) / (log_prob_max - log_prob_min)  # Normalize to [0, 1]

    # Plot the t-SNE output with colors based on log_prob
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=log_probs_normalized, cmap='viridis', alpha=0.7)
    scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=log_probs, cmap='viridis', alpha=0.7)
    plt.title("t-SNE visualization of latent vectors with log-probabilities")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.colorbar(scatter)  # Show color bar for the log_prob values
    plt.grid(True)
    
    
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved t-SNE visualization to {save_path}")

def visualize_latent_with_hist(latent_vectors, log_probs, save_path=None):
    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(latent_vectors)

    # Create figure and gridspec layout
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 0.3, 1.5], wspace=0.4)

    # t-SNE scatter plot
    ax0 = plt.subplot(gs[0])
    sc = ax0.scatter(z_tsne[:, 0], z_tsne[:, 1], c=log_probs, cmap='viridis', alpha=0.7)
    ax0.set_title("t-SNE visualization of latent vectors with log-probabilities")
    ax0.set_xlabel("t-SNE component 1")
    ax0.set_ylabel("t-SNE component 2")
    ax0.grid(True)

    # Colorbar
    ax1 = plt.subplot(gs[1])
    cb = plt.colorbar(sc, cax=ax1)
    cb.set_label("log-probability")
    ax1.yaxis.tick_left() 

    # Histogram
    ax2 = plt.subplot(gs[2])
    ax2.hist(log_probs, bins=30, orientation='horizontal', color='gray', edgecolor='black')
    ax2.set_xlabel("Count")
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved t-SNE visualization to {save_path}")
    plt.show()

def plot_log_prob_histogram(log_probs, bins=50, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.hist(log_probs, bins=bins, density=True, color='skyblue', edgecolor='black')
    plt.title("Histogram of log-probabilities")
    plt.xlabel("log-probability")
    plt.ylabel("Density")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":

    config_path = "/home/han/projects/robomimic/bc_trained_models/test_can/20250520121344/config.json"
    ckpt_path = "/home/han/projects/robomimic/bc_trained_models/test_can/20250520121344/models/model_epoch_350.pth"
    dataset_path = "/home/han/projects/robomimic/datasets/can/ph/low_dim_v141.hdf5"
    save_path = "/home/han/projects/robomimic/bc_trained_models/tsne_visualization/can_radial_350.png"

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    model = FileUtils.model_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    data_loader = get_data_loader(dataset_path=dataset_path)
  
    latent_vectors, log_probs = run_encoder(model, data_loader)

    # visualize_latent_space(latent_vectors, log_probs,save_path=save_path)

    # plot_log_prob_histogram(log_probs, bins=50, save_path=None)

    visualize_latent_with_hist(latent_vectors, log_probs,save_path=save_path)