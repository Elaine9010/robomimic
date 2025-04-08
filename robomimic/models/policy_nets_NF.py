"""

Contains torch Modules for policy networks. These networks take an
observation dictionary as input (and possibly additional conditioning,
such as subgoal or goal dictionaries) and produce action predictions,
samples, or distributions as outputs. Note that actions
are assumed to lie in [-1, 1], and most networks will have a final
tanh activation to help ensure this range.
"""
import textwrap
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from robomimic.models.policy_nets import ActorNetwork, GaussianActorNetwork

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.base_nets import Module
from robomimic.models.transformers import GPT_Backbone
from robomimic.models.obs_nets import MIMO_MLP, RNN_MIMO_MLP, MIMO_Transformer, ObservationDecoder
from robomimic.models.vae_nets import VAE
from robomimic.models.distributions import TanhWrappedDistribution


class NatPNActorNetwork(nn.Module):
    """
    Variant of the actor network that uses Natural Posterior Network (NatPN) 
    for uncertainty-aware action prediction.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        encoder="image-deep",
        flow="radial",
        flow_num_layers=8,
        certainty_budget="normal",
        dropout=0.1,
        learning_rate=1e-3,
        entropy_weight=1e-5,
        finetune=True,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): Maps modality to expected observation shapes.

            ac_dim (int): Dimension of action space.

            mlp_layer_dims ([int]): Sequence of integers for MLP hidden layer sizes.

            encoder (str): Encoder type for NatPN.

            flow (str): Type of normalizing flow.

            flow_num_layers (int): Number of flow layers.

            certainty_budget (str): Certainty budget for scaling log-probabilities.

            dropout (float): Dropout rate.

            learning_rate (float): Learning rate.

            entropy_weight (float): Weight for entropy regularizer.

            finetune (bool): Whether to fine-tune after main training.

            goal_shapes (OrderedDict): Expected shapes for goal observations.

            encoder_kwargs (dict or None): Encoder configurations.
        """
        super(NatPNActorNetwork, self).__init__()

        self.ac_dim = ac_dim
        self.obs_shapes = obs_shapes
        self.goal_shapes = goal_shapes if goal_shapes is not None else OrderedDict()

