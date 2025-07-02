import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces  # Use gymnasium for compatibility with SB3
from typing import Dict, Tuple, List, Any, Optional, Union

from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class TradingLSTMPolicy(RecurrentActorCriticPolicy):
    """
    Custom LSTM policy designed for trading environments.
    
    Inherits from SB3's RecurrentActorCriticPolicy and incorporates a trading-specific architecture.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: callable,
        *args,
        **kwargs
    ):
        # Define trading-specific parameters
        self.lstm_hidden_size = kwargs.pop("lstm_hidden_size", 64)
        self.feature_extractor_dim = kwargs.pop("feature_extractor_dim", 128)
        
        # Include custom network architecture parameters
        kwargs["lstm_hidden_size"] = self.lstm_hidden_size
        
        # Call the parent class initializer
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        # Output the architecture details
        print(f"Initialized TradingLSTMPolicy with LSTM size: {self.lstm_hidden_size}")


# Factory function to create the policy
def make_trading_lstm_policy():
    # Return the TradingLSTMPolicy class (do not instantiate it here)
    return TradingLSTMPolicy
