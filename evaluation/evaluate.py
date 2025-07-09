import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

from environment.lob_env import LOBEnv
from agent.policy import TradingLSTMPolicy

def evaluate(features_path: str,
             checkpoint_path: str,
             metrics_out: str = "metrics.json",
             logs_out: str = "step_logs.csv",
             states_out: str = "hidden_states.npy",
             deterministic: bool = True,
             max_eval_steps: int = None) -> pd.DataFrame:
    """
    Run one evaluation episode with a trained PPO agent against LOBEnv,
    aggregate per-step data (including hidden states), summary metrics,
    and return as a pandas DataFrame.

    Returns:
      df: DataFrame with one row per step, columns for all info fields and hidden_state
    """

    env = LOBEnv(features_csv=features_path)

    # 2) Load model manually
    model = PPO.load(checkpoint_path, device="cpu")

    # If using a custom policy, ensure model.policy is set or imported
    # e.g., model.policy = TradingLSTMPolicy

    # 3) Reset environment
    obs, _ = env.reset()
    total_steps = getattr(env, 'max_steps', None)
    if max_eval_steps == None:
        eval_steps = total_steps
    else:
        eval_steps = int(min(total_steps, max_eval_steps))
    pbar = tqdm(total=eval_steps, desc="Evaluating", unit="step")

    done = False
    lstm_state = None
    step_idx = 0
    episode_id = 0
    max_pnl = -np.inf

    # 4) Collect per-step records
    records = []
    hidden_states = []

    while not done and step_idx < eval_steps:
        # Get action (and new LSTM state if recurrent)
        try:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                deterministic=deterministic
            )
        except TypeError:
            # Non-recurrent policy: no 'state' kwarg
            action, _ = model.predict(obs, deterministic=deterministic)
            lstm_state = None

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)

        # Compute PnL and drawdown
        pnl = info['realized_pnl'] + info['unrealized_pnl']
        max_pnl = max(max_pnl, pnl)
        drawdown = max_pnl - pnl

        # Prepare record
        rec = {
            'episode_id':       episode_id,
            'step_idx':         step_idx,
            'timestamp':        info.get('timestamp'),
            'mid_price':        info.get('mid_price'),
            'spread':           info.get('spread'),
            'depth1_bid':       info.get('bid_price'),
            'depth1_ask':       info.get('ask_price'),
            'volume_imbalance': info.get('volume_imbalance'),
            'price_velocity':   info.get('price_velocity'),
            'action':           action,
            'reward':           reward,
            'pnl':              pnl,
            'drawdown':         drawdown,
            'inventory':        info.get('inventory'),
            'done':             done
        }
        records.append(rec)

        # Include hidden state if available
        if lstm_state is not None:
            h, c = lstm_state
            if isinstance(h, torch.Tensor):
                h_np = h.detach().cpu().numpy().flatten()
            else:
                h_np = np.array(h).flatten()
            if isinstance(c, torch.Tensor):
                c_np = c.detach().cpu().numpy().flatten()
            else:
                c_np = np.array(c).flatten()
            hidden_states.append(np.concatenate([h_np, c_np]))

        step_idx += 1
        pbar.update(1)

    pbar.close()

    # 5) Build DataFrame
    df = pd.DataFrame(records)

    # 6) Save summary metrics
    metrics = {
        'pnl_curve':      df['pnl'].tolist(),
        'drawdown':       df['drawdown'].tolist(),
        'spread_capture': df['spread'].tolist()
    }
    with open(metrics_out, 'w') as mf:
        json.dump(metrics, mf)

    # 8) Save full step logs
    df.to_csv(logs_out, index=False)

    hidden_array = np.vstack(hidden_states) if hidden_states else np.empty((0,))

    return df, hidden_array