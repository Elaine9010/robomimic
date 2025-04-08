# This script is used for training a BC-NatPN using the Robomimic library. And to visualize the training and validation loss.

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader

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

class PrintLogger(object):
    """
    This class redirects print statements to both console and a file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # ensure stdout gets flushed
        self.terminal.flush()

def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.
    """
    # Create training dataset and loader
    train_dataset = SequenceDataset(
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
    train_data_loader = DataLoader(
        dataset=train_dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )

    # Create validation dataset and loader
    val_dataset = SequenceDataset(
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
        filter_by_attribute="valid",       # can optionally provide a filter key here
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )

    return train_data_loader, val_data_loader


def get_example_model(config, dataset_path, device, use_natpn=True):
    """
    Use a default config to construct a BC model.
    """
    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)
    # read dataset to get some metadata for constructing model
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
                    dataset_path=dataset_path, 
                    all_obs_keys=sorted(( "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object" )))
    if use_natpn:
        # make BC-NatPN model
        model = BC_NatPN(
            algo_config=config.algo,
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )
        print("BC-NatPN model created")
        return model
    else:
        # make BC model
        print("Using BC model")
        model = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device=device,
        )
    return model


def print_batch_info(batch):
    print("\n============= Batch Info =============")
    for k in batch:
        if k in ["obs", "next_obs"]:
            print("key {}".format(k))
            for obs_key in batch[k]:
                print("    obs key {} with shape {}".format(obs_key, batch[k][obs_key].shape))
        else:
            print("key {} with shape {}".format(k, batch[k].shape))
    print("")


def run_epoch(model, data_loader,validation,grad_steps):
    """
    Args:
        model (Algo instance): instance of Algo class to use for training
        data_loader (torch.utils.data.DataLoader instance): torch DataLoader for
            sampling batches
        validation (bool): whether to run validation or training
        grad_steps (int): number of gradient steps per epoch
    Returns:
        float: average loss over all batches
    """
    has_printed_batch_info = False
    if validation:
        model.set_eval()
    else:
        model.set_train()

    # set up data loader iterator
    data_loader_iter = iter(data_loader)
    step_log_all = []
    for _ in range(grad_steps):
        # load next batch from data loader
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            # data loader ran out of batches - reset and yield first batch
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        # if not has_printed_batch_info:
        #     has_printed_batch_info = True
            # print_batch_info(batch)

        # process batch for training
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

        # forward and backward pass
        info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=validation)

        # record loss
        step_log = model.log_info(info)
        step_log_all.append(step_log)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = {
        k: float(np.mean([x.cpu().item() if hasattr(x, "cpu") else x for x in v]))
        for k, v in step_log_dict.items()
    }

    return step_log_all

def warmup_flow(model, data_loader, device, warmup_epochs=3):
    
    model.set_train()

    flow = model.nets["policy"].flow
    encoder = model.nets["policy"].encoder
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    for epoch in range(warmup_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
                # process batch for training
            input_batch = model.process_batch_for_training(batch)
            input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)
            obs = model.nets["encoder"](
                obs=input_batch["obs"],
                goal = input_batch.get("goal_obs", None),
            )
            with torch.no_grad():  # freeze encoder
                z = obs.detach().float()
            log_prob = flow(z)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"[Warm-up] Epoch {epoch + 1} - NLL: {avg_loss:.4f}")

def plot_losses(train_losses, epochs, val_losses, val_epochs,save_path=None):
    """
    Plot training and validation losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', linewidth=1.5, label='Training Loss')
    plt.plot(val_epochs, val_losses, marker='s', linestyle='--', linewidth=2, label='Validation Loss', color='orange')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Horizontal line at y=0
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")

def run_rollout(config, env_meta, shape_meta,rollout_model, envs, epoch):
    
    if True:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]
        print("env names: ", env_names)
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name, 
                render=False, 
                render_offscreen= True,
                use_image_obs=shape_meta["use_images"],
                use_depth_obs=shape_meta["use_depths"],
            )
            env = EnvUtils.wrap_env_from_config(env, config=config) 
            envs[env.name] = env

    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=envs,
        horizon=config.experiment.rollout.horizon,
        use_goals=config.use_goals,
        num_episodes=config.experiment.rollout.n,
        render=False,
        video_dir=video_dir if config.experiment.render_video else None,
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
    )
    return all_rollout_logs, video_paths


if __name__ == "__main__":

    # set up paths and parameters
    task_name = "square"
    warmup_epochs = None
    
    loss_save_path = "/home/han/projects/robomimic/bc_trained_models/losses"
    config_path = "/home/han/projects/robomimic/robomimic/exps/templates/bc_natpn.json"
    dataset_path = f"/home/han/projects/robomimic/datasets/{task_name}/ph/low_dim_v141.hdf5"
    # video_dir = "/home/han/projects/robomimic/bc_trained_models/videos"
    save_model = True
    model_save_path = f"/home/han/projects/robomimic/bc_trained_models/test_{task_name}/{task_name}_natpn"
    os.makedirs(model_save_path, exist_ok=True)

    # default BC config
    config = config_factory(algo_name="bc")  #TODO:change bc to bc_natpn
    ext_cfg = json.load(open(config_path, "r"))
    with config.values_unlocked():
        config.update(ext_cfg)
    print("Using config: ", config) 
    # Set seeds
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    num_epochs = config.train.num_epochs 
    grad_steps = config.experiment.epoch_every_n_steps
    rollout_every_n_epochs = config.experiment.rollout.rate
    validation_every_n_epochs = config.experiment.validation_epoch_every_n_steps
    print("Number of epochs: ", num_epochs)
    print("Gradient steps per epoch: ", grad_steps)
    print("Rollout every n epochs: ", rollout_every_n_epochs)
    print("Validation every n epochs: ", validation_every_n_epochs)
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # get dataset loader
    train_data_loader, val_data_loader = get_data_loader(dataset_path=dataset_path)

    # create model
    model = get_example_model(config = config, dataset_path=dataset_path, device=device, use_natpn=True)

    # create environment
    
    config.train.data = dataset_path
    envs = OrderedDict()
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    all_obs_keys = ['robot0_eef_pos', 'robot0_gripper_qpos', 'object', 'robot0_eef_quat']
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=all_obs_keys,
        verbose=True
    )

    # warm-up training
    if warmup_epochs is not None:
        print(f"\n[Warm-up] Training normalizing flow for {warmup_epochs} epochs...")
        warmup_flow(model, train_data_loader, device=device, warmup_epochs=warmup_epochs)

    
    best_valid_loss = None
    best_success_rate = None
    best_return = None
    last_ckpt_time = time.time()
    train_losses = []
    val_losses = []
    val_epochs = []

    # Main training loop
    for epoch in range(1, num_epochs + 1): # epoch numbers start at 1
        step_log = run_epoch(
            model=model, 
            data_loader=train_data_loader,
            validation=False,
            grad_steps=grad_steps
            )
        model.on_epoch_end(epoch)
        train_epoch_loss = step_log["Loss"]
        train_losses.append(train_epoch_loss)
        print("Train Epoch {}: Loss {}".format(epoch, train_epoch_loss))
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # run evaluation every n epochs
        if config.experiment.validate and epoch % validation_every_n_epochs == 0:
            steo_log = run_epoch(model=model, data_loader=val_data_loader,validation=True,grad_steps=grad_steps)
            val_losses.append(step_log["Loss"])
            val_epochs.append(epoch)
            print("Validation Loss {}".format(step_log["Loss"]))

        # run rollouts every n epochs
        if config.experiment.rollout.enabled and epoch % rollout_every_n_epochs == 0:
            rollout_model = RolloutPolicy(model)
            all_rollout_logs, video_paths = run_rollout(
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                rollout_model=rollout_model,
                envs=envs,
                epoch=epoch,
            )
            # Summarize results from rollouts to terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        # data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                        print("Timing_Stats/Rollout_{}_{}: {}".format(env_name, k[5:], v))
                    else:
                        # data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)
                        print("Rollout/{}/{}: {}".format(k, env_name, v))

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

        # save model and configs and logs
        if save_model and epoch % config.experiment.save.every_n_epochs == 0:
            # save_on_epoch = num_epochs
            # model_save_path =  f"{model_save_path}/models/model_epoch_{save_on_epoch}.pth"
            # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            # torch.save(model.serialize(), model_save_path)
            # print(f"Model saved to {model_save_path}")
            TrainUtils.save_model(
                    model=model,
                    config=config,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                    obs_normalization_stats=None,
                )
            

    epochs = list(range(1, num_epochs + 1))
    # plot training and validation losses
    save_path = f"{loss_save_path}/natpn_loss_{task_name}_{num_epochs}_epochs_warmup_{warmup_epochs}_radial_flow.png"
    plot_losses(train_losses, epochs, val_losses, val_epochs, save_path)

            

    

