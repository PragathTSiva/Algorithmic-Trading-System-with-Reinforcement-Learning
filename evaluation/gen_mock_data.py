import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_data(episodes, steps_per_episode, base_start_time):
    records = []

    for ep in range(1, episodes + 1):
        strategy = "momentum" if ep <= 5 else "mean_reversion"
        
        prices = []
        price = 100.0
        if strategy == "momentum":
            # trending random walk
            for _ in range(steps_per_episode):
                price += np.random.randn() * 0.05
                prices.append(price)
        else:
            # mean-reversion
            mu, theta = 100.0, 0.1
            for _ in range(steps_per_episode):
                price += theta * (mu - price) + np.random.randn() * 0.1
                prices.append(price)
        
        prices = np.array(prices)
        spreads = 0.05 + np.random.randn(steps_per_episode) * 0.005
        depth1_bid = prices - spreads / 2 + np.random.randn(steps_per_episode) * 0.01
        depth1_ask = prices + spreads / 2 + np.random.randn(steps_per_episode) * 0.01
        volume_imbalance = np.random.uniform(-1, 1, steps_per_episode)
        price_velocity = np.concatenate([[0], np.diff(prices)])
        
        threshold = 0.02
        actions = []
        for v in price_velocity:
            if abs(v) < threshold:
                actions.append(0)
            else:
                if strategy == "momentum":
                    actions.append(1 if v > 0 else 2)
                else:
                    actions.append(2 if v > 0 else 1)
        actions = np.array(actions)
        
        rand = np.random.rand(steps_per_episode)
        actions[rand < 0.05] = 3
        actions[(rand >= 0.05) & (rand < 0.1)] = 4
        
        # compute rewards and performance
        rewards = price_velocity * np.where(actions == 1, 1, np.where(actions == 2, -1, 0)) \
                - 0.05 * (actions != 0)
        pnl = np.cumsum(price_velocity * np.where(actions == 1, 1, np.where(actions == 2, -1, 0)))
        inventory = np.cumsum(np.where(actions == 1, 1, np.where(actions == 2, -1, 0)))
        done = [False] * (steps_per_episode - 1) + [True]
        
        # add hidden states
        if strategy == "momentum":
            hidden_states = np.random.normal(loc=0.0, scale=0.5, size=(steps_per_episode, 4))
        else:
            hidden_states = np.random.normal(loc=1.0, scale=0.7, size=(steps_per_episode, 4))
        
        timestamps = [
            base_start_time + timedelta(days=(ep - 1), minutes=i)
            for i in range(steps_per_episode)
        ]
        
        for i in range(steps_per_episode):
            records.append({
                "episode_id": ep,
                "step_idx": i,
                "timestamp": timestamps[i],
                "strategy": strategy,
                "mid_price": prices[i],
                "spread": spreads[i],
                "depth1_bid": depth1_bid[i],
                "depth1_ask": depth1_ask[i],
                "volume_imbalance": volume_imbalance[i],
                "price_velocity": price_velocity[i],
                "action": actions[i],
                "reward": rewards[i],
                "pnl": pnl[i],
                "inventory": inventory[i],
                "done": done[i],
                "hidden_state_0": hidden_states[i, 0],
                "hidden_state_1": hidden_states[i, 1],
                "hidden_state_2": hidden_states[i, 2],
                "hidden_state_3": hidden_states[i, 3],
            })

    return records

if __name__ == "__main__":
    episodes = 10
    steps_per_episode = 100
    base_start_time = datetime(2025, 5, 4, 9, 30)
    records = generate_data(episodes, steps_per_episode, base_start_time)

    df = pd.DataFrame(records)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(BASE_DIR, "mock_training_log.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Mock training log saved to {output_path}")