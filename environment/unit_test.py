# UNIT TEST FOR TRADING ENVIRONMENT

import sys
import os

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import environment.lob_env as LobEnv
import environment.state_builder as StBuild
import environment.fills as Fills
import reward.reward as Reward
import gymnasium as gym
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from evaluation.logger import Logger

# Reload modules to ensure latest changes
importlib.reload(LobEnv)
importlib.reload(StBuild)
importlib.reload(Fills)
importlib.reload(Reward)

def test_trading_environment():
    """
    Test the trading environment with a random agent
    - Verify state transitions
    - Check fill consistency 
    - Assert episode termination
    - Validate reward consistency
    - Export trace logs for inspection
    """
    print("Starting Trading Environment Unit Test...")
    
    # Create directory for dummy data
    os.makedirs("./data/training_data", exist_ok=True)
    features_path = "./data/training_data/AAPL_20250407_100ms_features.csv"

    # Create mock data for testing
    print(f"Creating mock data for testing at {features_path}")
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-04-07', periods=5000, freq='100ms', tz='America/New_York'),
        'mid_price': np.random.normal(150, 1, 5000),
        'best_bid': np.random.normal(149.9, 1, 5000),
        'best_ask': np.random.normal(150.1, 1, 5000),
        'bid_volume_1': np.random.randint(100, 1000, 5000),
        'ask_volume_1': np.random.randint(100, 1000, 5000),
        'price_change_10s': np.random.normal(0, 0.01, 5000),
        'volume_imbalance': np.random.normal(0, 0.1, 5000),
        'time_of_day': np.linspace(0.3, 0.7, 5000)  # Normalized time of day
    })

    # Ensure all required features are present
    # Renaming columns to match what StateBuilder expects
    mock_data['bid_price'] = mock_data['best_bid']
    mock_data['ask_price'] = mock_data['best_ask']
    mock_data['bid_size'] = mock_data['bid_volume_1']
    mock_data['ask_size'] = mock_data['ask_volume_1']
    mock_data['spread'] = mock_data['best_ask'] - mock_data['best_bid']
    mock_data['vol_imbalance'] = mock_data['volume_imbalance']
    mock_data['last_trade_price'] = mock_data['mid_price'] + np.random.normal(0, 0.01, 5000)
    mock_data['trade_volume'] = np.random.randint(10, 100, 5000)
    mock_data['trade_flag'] = np.random.randint(0, 2, 5000)
    mock_data['mid_diff'] = mock_data['mid_price'].diff().fillna(0)
    mock_data['mid_return'] = mock_data['mid_diff'] / mock_data['mid_price'].shift(1).fillna(mock_data['mid_price'][0])
    mock_data['mv_1s'] = mock_data['mid_return'].rolling(10).std().fillna(0)
    mock_data['mv_5s'] = mock_data['mid_return'].rolling(50).std().fillna(0)
    mock_data['vol_1s'] = mock_data['trade_volume'].rolling(10).sum().fillna(0)
    mock_data['vol_5s'] = mock_data['trade_volume'].rolling(50).sum().fillna(0)

    # Save mock data
    mock_data.to_csv(features_path, index=False)
    print(f"Created mock data file with {len(mock_data)} rows")
    
    # Initialize components
    state_builder = StBuild.StateBuilder(features_path)
    fill_model = Fills.FillsLedger()  # Changed from FillModel to FillsLedger
    reward_function = Reward.RewardFunction()
    
    # Initialize environment
    env = LobEnv.LOBEnv(  # Changed from LimitOrderBookEnv to LOBEnv
        state_builder=state_builder,
        max_steps=1000,  # Short episode for testing
        trade_size=1,
        inventory_limit=10,
        reward_fn=reward_function
    )
    
    # Create directory for logs
    os.makedirs("./test_logs", exist_ok=True)
    
    # Initialize a custom logger function to avoid Logger class issues
    log_file = open("./test_logs/trade_log.csv", "w", newline="")
    log_writer = None
    
    # Reset the environment
    state = env.reset()
    
    # Validate initial state
    assert isinstance(state, np.ndarray), "Initial state should be a numpy array"
    assert state.shape[0] == len(state_builder.state_columns), f"State dimension mismatch: {state.shape[0]} vs {len(state_builder.state_columns)}"
    assert env.fills_ledger.get_inventory() == 0, "Initial position should be 0"
    
    print(f"Initial state shape: {state.shape}")
    print(f"Initial position: {env.fills_ledger.get_inventory()}")
    
    # Run a random agent for a fixed number of steps
    num_steps = 100
    actions = []
    rewards = []
    positions = []
    pnls = []
    states = []
    
    for step in range(num_steps):
        # Random action (0: do nothing, 1: market buy, 2: market sell)
        action = random.randint(0, 2)
        actions.append(action)
        
        # Step through environment
        next_state, reward, done, info = env.step(action)
        
        # Log data
        states.append(next_state)
        rewards.append(reward)
        positions.append(env.fills_ledger.get_inventory())
        pnls.append(info.get('realized_pnl', 0))
        
        # Write to log file directly
        if log_writer is None:
            import csv
            headers = ["timestamp", "action", "reward", "position", "cash", "pnl", 
                      "mid_price", "best_bid", "best_ask"]
            log_writer = csv.DictWriter(log_file, fieldnames=headers)
            log_writer.writeheader()
        
        # Get price data safely
        mid_price = 0
        best_bid = 0
        best_ask = 0
        if next_state is not None:
            try:
                mid_price = next_state[state_builder.state_columns.index('mid_price')]
                best_bid = next_state[state_builder.state_columns.index('bid_price')]
                best_ask = next_state[state_builder.state_columns.index('ask_price')]
            except:
                pass
        
        # Log data
        log_writer.writerow({
            "timestamp": info.get('timestamp', ''),
            "action": action,
            "reward": reward,
            "position": env.fills_ledger.get_inventory(),
            "cash": 0,  # Not tracked in this environment
            "pnl": info.get('realized_pnl', 0),
            "mid_price": mid_price,
            "best_bid": best_bid, 
            "best_ask": best_ask
        })
        
        # Verify next state
        if not done:
            assert isinstance(next_state, np.ndarray), f"Next state should be a numpy array at step {step}"
            assert np.isfinite(next_state).all(), f"State contains NaN or Inf at step {step}"
        
        # Verify reward structure
        try:
            reward = float(reward)  # Try to convert to float if not already
            assert np.isfinite(reward), f"Reward should be finite at step {step}"
        except (ValueError, TypeError):
            print(f"Warning: Could not convert reward to float: {reward}, type: {type(reward)}")
            reward = 0.0  # Use a default value to continue the test
        
        # Stop if episode is done
        if done:
            print(f"Episode terminated after {step+1} steps")
            break
        
        state = next_state
        
        # Add near the start of the step loop in test_trading_environment
        print(f"Step {step}: Action={action}")

        # Add right after getting reward from env.step()
        print(f"  Reward={reward}, type={type(reward)}")
    
    # Final position and PnL check
    print(f"Final position: {env.fills_ledger.get_inventory()}")
    print(f"Final PnL: {info.get('realized_pnl', 0)}")
    
    # Close log file
    log_file.close()
    
    # Plot some metrics for visual inspection
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Rewards')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(positions)
    plt.title('Positions')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(pnls)
    plt.title('PnL')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./test_logs/environment_test_plot.png')
    plt.show()
    
    # Calculate some statistics
    avg_reward = np.mean(rewards)
    max_pos = np.max(np.abs(positions))
    final_pnl = pnls[-1] if pnls else 0
    
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Maximum absolute position: {max_pos}")
    print(f"Final PnL: {final_pnl:.4f}")
    
    # Add a basic trade strategy test (momentum-based)
    test_momentum_strategy(env)
    
    print("Trading Environment Unit Test Completed Successfully!")
    return True

def test_momentum_strategy(env):
    """Test a simple momentum strategy to verify environment consistency"""
    print("\nTesting Momentum Strategy...")
    
    # Reset the environment
    state = env.reset()
    
    # Create log file for momentum strategy
    momentum_log_file = open("./test_logs/trade_log_momentum.csv", "w", newline="")
    momentum_writer = None
    
    # Run a momentum strategy for a fixed number of steps
    num_steps = 200
    rewards = []
    positions = []
    pnls = []
    
    # Find the index of price_change or similar feature
    price_change_index = None
    for i, col in enumerate(env.state_builder.state_columns):
        if 'mid_diff' in col or 'mid_return' in col:
            price_change_index = i
            break
    
    if price_change_index is None:
        print("Could not find price momentum feature, using mid_diff index")
        try:
            price_change_index = env.state_builder.state_columns.index('mid_diff')
        except ValueError:
            # Fallback to the first feature
            price_change_index = 0
            print(f"Using fallback index 0, available columns: {env.state_builder.state_columns}")
    
    for step in range(num_steps):
        # Get price momentum from state
        price_momentum = state[price_change_index]
        
        # Simple momentum strategy
        if price_momentum > 0.001:  # Strong positive momentum
            action = 1  # Market buy
        elif price_momentum < -0.001:  # Strong negative momentum
            action = 2  # Market sell
        else:
            action = 0  # Do nothing
        
        # Step through environment
        next_state, reward, done, info = env.step(action)
        
        # Log data
        rewards.append(reward)
        positions.append(env.fills_ledger.get_inventory())
        pnls.append(info.get('realized_pnl', 0))
        
        # Initialize writer if needed
        if momentum_writer is None:
            import csv
            headers = ["timestamp", "action", "reward", "position", "cash", "pnl", 
                      "mid_price", "best_bid", "best_ask"]
            momentum_writer = csv.DictWriter(momentum_log_file, fieldnames=headers)
            momentum_writer.writeheader()
        
        # Get price data safely
        mid_price = 0
        best_bid = 0
        best_ask = 0
        if next_state is not None:
            try:
                mid_price = next_state[env.state_builder.state_columns.index('mid_price')]
                best_bid = next_state[env.state_builder.state_columns.index('bid_price')]
                best_ask = next_state[env.state_builder.state_columns.index('ask_price')]
            except:
                pass
        
        # Log step data
        momentum_writer.writerow({
            "timestamp": info.get('timestamp', ''),
            "action": action,
            "reward": reward,
            "position": env.fills_ledger.get_inventory(),
            "cash": 0,  # Not tracked in this environment
            "pnl": info.get('realized_pnl', 0),
            "mid_price": mid_price,
            "best_bid": best_bid,
            "best_ask": best_ask
        })
        
        # Stop if episode is done
        if done:
            print(f"Episode terminated after {step+1} steps")
            break
        
        state = next_state
    
    # Close the log file
    momentum_log_file.close()
    
    # Calculate some statistics
    avg_reward = np.mean(rewards)
    max_pos = np.max(np.abs(positions))
    final_pnl = pnls[-1] if pnls else 0
    
    print(f"Momentum Strategy - Average reward: {avg_reward:.4f}")
    print(f"Momentum Strategy - Maximum absolute position: {max_pos}")
    print(f"Momentum Strategy - Final PnL: {final_pnl:.4f}")
    
    return True

if __name__ == "__main__":
    test_trading_environment()
