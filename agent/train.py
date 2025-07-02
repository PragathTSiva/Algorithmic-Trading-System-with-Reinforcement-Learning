import os
from typing import Any, Dict, Callable
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.ppo_recurrent import RecurrentPPO
from agent.callbacks import EarlyStoppingCallback, CheckpointCallback
from agent.policy import make_trading_lstm_policy


def train_agent(
    env_fn: Callable[[], gym.Env],
    config: Dict[str, Any],
    log_dir: str = "./logs",
    save_path: str = "./models",
    verbose: int = 1
) -> RecurrentPPO:
    """
    Train a PPO agent with LSTM policy on vectorized environments.

    Args:
        env_fn: Factory that returns a new Gym environment instance.
        config: Training configuration parameters.
        log_dir: Directory to save logs.
        save_path: Directory to save models.
        verbose: Verbosity level.

    Returns:
        A trained RecurrentPPO model.
    """
    # Create necessary directories
    for dir_path in [log_dir, save_path]:
        os.makedirs(dir_path, exist_ok=True)

    # Vectorized environments: each worker gets a fresh instance
    n_envs = config.get("n_envs", 1)
    train_env = DummyVecEnv([lambda: Monitor(env_fn()) for _ in range(n_envs)])

    # Set up callbacks
    callbacks = []
    if config.get("save_checkpoints", True):
        callbacks.append(
            CheckpointCallback(
                save_freq=config.get("save_freq", 10000),
                save_path=save_path,
                name_prefix="model",
                verbose=verbose
            )
        )
    if config.get("early_stopping", False):
        callbacks.append(
            EarlyStoppingCallback(
                check_freq=config.get("check_freq", 1000),
                min_improvement=config.get("min_improvement", 0.01),
                patience=config.get("patience", 5),
                verbose=verbose
            )
        )

    # Choose policy
    use_custom_policy = config.get("use_custom_policy", False)
    policy = make_trading_lstm_policy() if use_custom_policy else "MlpLstmPolicy"

    # Initialize PPO with LSTM policy
    model = RecurrentPPO(
        policy,
        train_env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        tensorboard_log=log_dir,
        verbose=verbose
    )

    # Attach extra callbacks like progress bar
    extra_callbacks = config.get("extra_callbacks", [])
    all_callbacks = callbacks + extra_callbacks if callbacks or extra_callbacks else None

    # Begin learning
    model.learn(
        total_timesteps=config.get("total_timesteps", 100000),
        callback=all_callbacks
    )

    # Save final model
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    if verbose > 0:
        print(f"Model saved to {final_model_path}")

    return model