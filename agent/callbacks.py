from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """
    Simple early stopping callback that monitors mean reward
    and stops training if no improvement is seen.
    """
    def __init__(
        self, 
        check_freq: int = 1000,
        min_improvement: float = 0.01,
        patience: int = 5,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_improvement = min_improvement
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and len(self.model.ep_info_buffer) > 0:
            # Calculate mean episode reward
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            
            if self.verbose > 0:
                print(f"Mean reward: {mean_reward:.2f} (best: {self.best_mean_reward:.2f})")
            
            # Check for improvement
            if mean_reward > self.best_mean_reward * (1 + self.min_improvement):
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            # Stop training if no improvement for several checks
            if self.no_improvement_count >= self.patience and self.best_mean_reward != -np.inf:
                if self.verbose > 0:
                    print(f"Stopping training after {self.patience} checks without improvement")
                return False
                
        return True


class CheckpointCallback(BaseCallback):
    """
    Simple callback that saves model when mean reward improves.
    """
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./models",
        name_prefix: str = "model",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0 and len(self.model.ep_info_buffer) > 0:
            # Calculate mean reward
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            
            # Save model if improved
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                # Save the model
                model_path = f"{self.save_path}/{self.name_prefix}_{self.n_calls}_steps"
                self.model.save(model_path)
                
                if self.verbose > 0:
                    print(f"Saved model to {model_path} (reward: {mean_reward:.2f})")
                    
        return True
