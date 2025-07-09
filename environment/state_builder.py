# agitrader/environment/state_builder.py

import pandas as pd
import numpy as np

class StateBuilder:
    """
    Loads feature CSV and provides market state vectors for Gym environments.
    """
    def __init__(self, features_path: str):
        self.df = pd.read_csv(features_path, parse_dates=["timestamp"])
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], format="ISO8601")
        self.df.set_index("timestamp", inplace=True)
        self.df = self.df.tz_convert("America/New_York")

        # Normalize time of day to [0, 1] (market open = 09:30)
        market_open = pd.to_datetime(self.df.index.date[0].strftime("%Y-%m-%d") + " 09:30:00").tz_localize("America/New_York")
        seconds_in_day = 6.5 * 3600
        self.df["time_of_day"] = (self.df.index - market_open).total_seconds() / seconds_in_day

        # Drop any NaNs
        self.df.dropna(inplace=True)
        self.valid_indices = self.df.index

        # Define state vector features
        self.state_columns = [
            "bid_price", "ask_price", "bid_size", "ask_size",
            "mid_price", "spread", "vol_imbalance",
            "last_trade_price", "trade_volume", "trade_flag",
            "mid_diff", "mid_return", "mv_1s", "mv_5s", "vol_1s", "vol_5s",
            "time_of_day"
        ]

    def get_state(self, idx: int) -> np.ndarray:
        """
        Returns the state vector at row `idx` as a NumPy array.
        """
        if idx >= len(self.df):
            raise IndexError("Index out of bounds for StateBuilder.")
        return self.df.iloc[idx][self.state_columns].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)
