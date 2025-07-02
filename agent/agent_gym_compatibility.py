"""
Agent-Gym Compatibility Module

This module tests the compatibility between our RL agent and the LOBEnv Gym environment.
We'll build it step by step and test each component individually.
"""
import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces as gym_spaces

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after adding to path
from environment.lob_env import LOBEnv
from environment.state_builder import StateBuilder
from reward.reward import RewardFunction
from agent.train import train_agent

class LOBEnvGymWrapper(gym.Env):
    """
    Wrapper for LOBEnv to make it compatible with StableBaselines3 (gymnasium).
    This bridges the gap between gym and gymnasium interfaces.
    """
    def __init__(self, lob_env):
        self.env = lob_env
        
        # Convert gym spaces to gymnasium spaces
        self.action_space = gym_spaces.Discrete(self.env.action_space.n)
        self.observation_space = gym_spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs, {}  # Return observation and empty info dict for gymnasium compatibility
    
    def step(self, action):
        # Call the underlying environment's step function
        obs, reward, done, info = self.env.step(action)
        
        # In gymnasium, the step method returns 5 values:
        # obs, reward, terminated, truncated, info
        # We convert gym's done flag to terminated and set truncated as False
        return obs, reward, done, False, info
    
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()

def create_sample_features_csv(path="./data/sample_features.csv"):
    """Create a simple sample features CSV for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Generate 100 rows of sample market data
    n_rows = 100
    # Create localized timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_rows, freq='1min', tz='UTC')
    
    data = {
        'timestamp': timestamps,
        'mid_price': np.random.normal(100, 1, n_rows),
        'spread': np.abs(np.random.normal(0.1, 0.02, n_rows)),
        'bid_price': np.zeros(n_rows),
        'ask_price': np.zeros(n_rows),
        'bid_size': np.random.randint(1, 100, n_rows),
        'ask_size': np.random.randint(1, 100, n_rows),
        'volume': np.random.randint(100, 1000, n_rows),
        'price_change': np.random.normal(0, 0.05, n_rows),
    }
    
    # Calculate bid and ask prices from mid and spread
    data['bid_price'] = data['mid_price'] - data['spread'] / 2
    data['ask_price'] = data['mid_price'] + data['spread'] / 2
    
    # Add all required features that StateBuilder expects
    data['vol_imbalance'] = np.random.normal(0, 0.1, n_rows)
    data['last_trade_price'] = data['mid_price'] + np.random.normal(0, 0.01, n_rows)
    data['trade_volume'] = np.random.randint(10, 100, n_rows)
    data['trade_flag'] = np.random.randint(0, 2, n_rows)
    data['mid_diff'] = np.random.normal(0, 0.01, n_rows)
    data['mid_return'] = np.random.normal(0, 0.001, n_rows)
    data['mv_1s'] = np.random.normal(0, 0.001, n_rows)
    data['mv_5s'] = np.random.normal(0, 0.002, n_rows)
    data['vol_1s'] = np.random.randint(10, 1000, n_rows)
    data['vol_5s'] = np.random.randint(50, 5000, n_rows)
    data['time_of_day'] = np.linspace(0.3, 0.7, n_rows)
    
    # Create and save the DataFrame
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Created sample features at {path}")
    return path

def test_sample_data_creation():
    """Test that we can create sample data without errors."""
    try:
        file_path = create_sample_features_csv()
        assert os.path.exists(file_path), f"File {file_path} was not created"
        print("✓ Sample data creation test passed")
        return True
    except Exception as e:
        print(f"✗ Sample data creation test failed: {e}")
        return False

def test_env_creation():
    """Test that we can create a LOBEnv instance and it has the expected interface."""
    try:
        # Create sample data
        features_path = create_sample_features_csv()
        
        # Create environment
        env = LOBEnv(
            features_csv=features_path,
            max_steps=50,
            inventory_limit=5,
            hold_cost_coeff=0.01,
            trade_size=1
        )
        
        # Check observation and action spaces
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Verify action space is correct (Discrete(3))
        assert env.action_space.n == 3, f"Expected action space to be Discrete(3), got {env.action_space}"
        
        print("✓ Environment creation test passed")
        return env
    except Exception as e:
        print(f"✗ Environment creation test failed: {e}")
        return None

def test_env_reset_step(env):
    """Test environment reset and step functions."""
    try:
        # Test reset
        obs = env.reset()
        print(f"Reset observation shape: {obs.shape}")
        assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
        assert obs.shape == env.observation_space.shape, "Observation shape should match space"
        
        # Test step with each possible action
        for action in range(3):  # 0=hold, 1=buy, 2=sell
            env.reset()  # Start fresh for each action
            next_obs, reward, done, info = env.step(action)
            
            print(f"Action {action} resulted in: reward={reward:.4f}, done={done}")
            print(f"  Info keys: {list(info.keys())}")
            
            # Verify observation format
            assert isinstance(next_obs, np.ndarray) or next_obs is None, f"Invalid observation type: {type(next_obs)}"
            if not done and next_obs is not None:
                assert next_obs.shape == env.observation_space.shape, "Observation shape should match space"
            
            # Verify info contains action_mask
            assert 'action_mask' in info, "Info dict should contain action_mask"
            assert len(info['action_mask']) == 3, "Action mask should have 3 elements for Discrete(3)"
            
            # Print the action mask
            print(f"  Action mask: {info['action_mask']}")
        
        print("✓ Environment reset/step test passed")
        return True
    except Exception as e:
        print(f"✗ Environment reset/step test failed: {e}")
        return False

def run_random_episode(env, max_steps=None):
    """Run a single episode with random actions, respecting action masks."""
    try:
        # Reset the environment
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Track episode data
        actions = []
        rewards = []
        positions = []
        pnls = []
        action_masks = []
        
        # First action is always valid hold (0) to start the episode
        action = 0
        
        while not done:
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Record data
            actions.append(action)
            rewards.append(reward)
            positions.append(info.get('inventory', 0))
            pnls.append(info.get('realized_pnl', 0))
            
            # Update for next iteration
            total_reward += reward
            steps += 1
            state = next_state
            
            print(f"Step {steps}: Action={action}, Reward={reward:.4f}, Position={info.get('inventory', 0)}")
            
            # Get action mask for next step
            if not done:
                action_mask = info.get('action_mask', [True, True, True])
                action_masks.append(action_mask)
                
                # Sample valid action using the mask
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    print("Warning: No valid actions available, defaulting to action 0 (hold)")
                    action = 0
            
            if max_steps and steps >= max_steps:
                print(f"Reached max steps ({max_steps})")
                break
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(actions)
        plt.title('Actions (0=hold, 1=buy, 2=sell)')
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        plt.plot(rewards)
        plt.title('Rewards')
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        plt.plot(positions)
        plt.title('Inventory Position')
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot(pnls)
        plt.title('Realized PnL')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Create logs directory
        os.makedirs("./logs/random_episodes", exist_ok=True)
        plt.savefig('./logs/random_episodes/random_episode_results.png')
        
        print("✓ Random episode test passed")
        return {
            'steps': steps,
            'total_reward': total_reward,
            'final_inventory': info.get('inventory', 0),
            'final_pnl': info.get('realized_pnl', 0),
            'actions': actions,
            'rewards': rewards,
            'positions': positions,
            'pnls': pnls,
            'action_masks': action_masks
        }
    except Exception as e:
        print(f"✗ Random episode test failed: {e}")
        return None

def test_agent_training(env):
    """Test agent training on the LOBEnv environment."""
    try:
        print("\nTesting agent training on LOBEnv...")
        
        # Create a wrapper around the LOBEnv to make it compatible with StableBaselines3
        wrapped_env = LOBEnvGymWrapper(env)
        
        # Setup minimal training config for testing purposes
        config = {
            "total_timesteps": 1000,  # Short training run for testing
            "n_envs": 1,
            "learning_rate": 3e-4,
            "n_steps": 50,
            "batch_size": 16,
            "use_custom_policy": True
        }
        
        # Set up temporary directories
        log_dir = "./logs/agent_gym_test"
        model_dir = "./models/agent_gym_test"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Train agent
        model = train_agent(
            env=wrapped_env, 
            config=config,
            log_dir=log_dir,
            save_path=model_dir,
            verbose=1
        )
        
        # Test predictions
        print("\nTesting trained model predictions...")
        obs, _ = wrapped_env.reset()
        lstm_states = None
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        
        print(f"Model predicted action: {action}")
        
        # Test step with predicted action
        next_obs, reward, done, _, info = wrapped_env.step(action)
        print(f"Step result: reward={reward:.4f}, done={done}")
        print(f"New position: {info.get('inventory', 0)}")
        
        print("✓ Agent training/prediction test passed")
        return True
    except Exception as e:
        print(f"✗ Agent training/prediction test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting agent-gym compatibility tests...")
    
    # Test 1: Sample data creation
    if not test_sample_data_creation():
        print("Stopping tests due to failure in sample data creation")
        sys.exit(1)
    
    # Test 2: Environment creation
    env = test_env_creation()
    if env is None:
        print("Stopping tests due to failure in environment creation")
        sys.exit(1)
    
    # Test 3: Environment reset and step
    if not test_env_reset_step(env):
        print("Stopping tests due to failure in environment reset/step test")
        sys.exit(1)
    
    # Test 4: Run a full random episode
    results = run_random_episode(env, max_steps=30)
    if results is None:
        print("Stopping tests due to failure in random episode test")
        sys.exit(1)
    
    # Test 5: Agent training and prediction
    if not test_agent_training(env):
        print("Stopping tests due to failure in agent training test")
        sys.exit(1)
    
    print("\nAll agent-gym compatibility tests passed!")
    print(f"Random episode results: {results['steps']} steps, total reward: {results['total_reward']:.4f}")
    print(f"Final inventory: {results['final_inventory']}, Final PnL: {results['final_pnl']:.4f}")
    print("\nAgent has been successfully integrated with LOBEnv environment!") 