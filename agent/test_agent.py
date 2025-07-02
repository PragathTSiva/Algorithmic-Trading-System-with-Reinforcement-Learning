import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.ppo_recurrent import RecurrentPPO

from train import train_agent


# Simple placeholder environment
class SimpleTradingEnv(gym.Env):
    """Minimal trading environment for testing the agent pipeline."""
    
    def __init__(self, obs_dim=10, episode_length=100):
        super().__init__()
        self.obs_dim = obs_dim
        self.episode_length = episode_length
        self.current_step = 0
        
        # Action space: buy (0), sell (1), hold (2), quote (3)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: vector of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}
    
    def step(self, action):
        # Simple reward logic: buy/sell actions have more variance
        if action in [0, 1]:  # buy or sell
            reward = np.random.normal(0.1, 0.5)
        else:  # hold or quote
            reward = np.random.normal(0, 0.1)
        
        self.current_step += 1
        done = self.current_step >= self.episode_length
        truncated = False
        info = {}
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self):
        # Generate random market features
        return np.random.randn(self.obs_dim).astype(np.float32)


if __name__ == "__main__":
    # Create and validate the environment
    env = SimpleTradingEnv(obs_dim=10, episode_length=100)
    check_env(env)
    print("Environment validation passed!")
    
    # Define training configuration
    config = {
        "total_timesteps": 20000,  # Short training run for testing
        "n_envs": 2,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 128,
        "early_stopping": True,
        "check_freq": 1000,
        "patience": 3,
        "save_checkpoints": True,
        "save_freq": 5000,
        "use_custom_policy": True,  # Set to False to use default MlpLstmPolicy
    }
    
    # Set up directories
    log_dir = "./logs/test_run"
    model_dir = "./models/test_run"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Train the agent
    print("\nStarting agent training...")
    model = train_agent(
        env=env,
        config=config,
        log_dir=log_dir,
        save_path=model_dir,
        verbose=1
    )
    
    # Test loading the saved model
    final_model_path = os.path.join(model_dir, "final_model.zip")
    loaded_model = RecurrentPPO.load(final_model_path)
    print(f"\nSuccessfully loaded model from {final_model_path}")

    print("\nTest completed successfully!")
    
    # Try a prediction with the model
    obs = env.reset()[0]
    lstm_states = None  # For recurrent policies, we need to track LSTM state
    
    # Get a prediction from the model
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    print(f"\nModel predicted action: {action}") 