import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.state_builder import StateBuilder
from environment.fills import FillsLedger
from reward.reward_GRPO import RewardGRPO
from evaluation.logger import Logger

DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "training_logs")
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

class LOBEnv(gym.Env):
    """
    Limit Order Book Gym Environment integrated with external RewardFunction.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        *,
        features_csv=None,
        state_builder=None,
        max_steps=None,
        inventory_limit=np.inf,
        hold_cost_coeff=0.0,
        trade_size=1,
        reward_fn = None,
        log_path: str = None  # new: allow custom log path per env
    ):
        # Initialize StateBuilder
        if state_builder is not None:
            self.state_builder = state_builder
        elif features_csv is not None:
            self.state_builder = StateBuilder(features_csv)
        else:
            raise ValueError("Provide either features_csv or state_builder")

        self.state_columns    = self.state_builder.state_columns
        self.timestamps       = self.state_builder.valid_indices
        self.max_steps        = max_steps or len(self.timestamps)
        self.inventory_limit  = inventory_limit
        self.hold_cost_coeff  = hold_cost_coeff
        self.trade_size       = trade_size

        # Initialize FillsLedger
        self.fills_ledger = FillsLedger()

        # Initialize RewardFunction
        self.reward_fn = reward_fn or RewardGRPO(
            pnl_weight=1.5,
            inventory_risk_weight=self.hold_cost_coeff,
            drawdown_weight=0.05,
            normalize_rewards=True
        )

        self.reward_fn.reset()

        # Logging setup
        headers = [
            'episode_id','step_idx','timestamp','mid_price',
            'spread','depth1_bid','depth1_ask','volume_imbalance',
            'price_velocity','action','reward','pnl','inventory','done'
        ]
        # Use provided log_path or default
        if log_path is None:
            log_path = os.path.join(DEFAULT_LOG_DIR, "training_log.csv")
        self.logger = Logger(log_path, headers=headers)

        # a small state‐variable to track episodes
        self.current_episode = 0

        # Action space: 0=hold,1=buy,2=sell
        self.action_space = spaces.Discrete(3)

        # Observation space matches feature vector length
        n_features = len(self.state_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # Internal pointers/state
        self.current_step = 0

        # History log for rendering / callbacks
        self.trade_history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # needed if implement RNG
        self.current_step = 0
        self.current_episode += 1
        self.fills_ledger.reset()
        self.reward_fn.reset()
        self.trade_history.clear()
        obs = self.state_builder.get_state(self.current_step)
        return obs, {}  # Gymnasium expects 2-tuple: (obs, info)

    def step(self, action):
        obs = self.state_builder.get_state(self.current_step)
        ts  = self.timestamps[self.current_step]

        # Process market orders
        price = None
        if action == 1 and self._can_buy():
            price = obs[self.state_columns.index('ask_price')]
            self.fills_ledger.add_fill(ts, price, self.trade_size, 'buy')
        elif action == 2 and self._can_sell():
            price = obs[self.state_columns.index('bid_price')]
            self.fills_ledger.add_fill(ts, price, self.trade_size, 'sell')

        # Advance time pointer
        self.current_step += 1
        done = (
            self.current_step >= self.max_steps or
            abs(self.fills_ledger.get_inventory()) > self.inventory_limit
        )

        next_obs = None if done else self.state_builder.get_state(self.current_step)

        # Gather metrics for reward calculation
        inventory       = self.fills_ledger.get_inventory()
        mid_price       = obs[self.state_columns.index('mid_price')]
        spread          = obs[self.state_columns.index('spread')]
        pnl_realized    = self.fills_ledger.compute_realized_pnl()
        pnl_unrealized  = self.fills_ledger.compute_unrealized_pnl(mid_price)

        # Compute composite reward via external RewardFunction
        reward = self.reward_fn.calculate_reward(
            action=action,
            position=inventory,
            pnl_realized=pnl_realized,
            pnl_unrealized=pnl_unrealized,
            spread=spread,
            executed_price=price,
            mid_price=mid_price,
            order_type='limit' if action in [1, 2] else 'market',
            ms_since_order=0.0,  # or real latency if available
            depth_consumed=self.trade_size,
            depth_total=obs[self.state_columns.index('depth_total')] if 'depth_total' in self.state_columns else 1.0,
            market_signals=None,  # or pass actual dict later
            done=done
        )

        # Info dict for diagnostics
        info = {
            'timestamp':      ts,
            'inventory':      inventory,
            'mid_price':      mid_price,
            'spread':         spread,
            'bid_price':      obs[self.state_columns.index('bid_price')],
            'ask_price':      obs[self.state_columns.index('ask_price')],
            'volume_imbalance': obs[self.state_columns.index('vol_imbalance')],
            'price_velocity': obs[self.state_columns.index('mid_diff')], 
            'realized_pnl':   pnl_realized,
            'unrealized_pnl': pnl_unrealized,
            'action_mask':    self._action_mask(),
            **self.reward_fn.get_reward_components()
        }

        # Log step data using Logger
        self.logger.log_step(
            episode_id       = self.current_episode,
            step_idx         = self.current_step,
            timestamp        = ts,
            mid_price        = mid_price,
            spread           = spread,
            depth1_bid       = info['bid_price'],
            depth1_ask       = info['ask_price'],
            volume_imbalance = info['volume_imbalance'],
            price_velocity   = info['price_velocity'],
            action           = action,
            reward           = reward,
            pnl              = pnl_realized + pnl_unrealized,
            inventory        = inventory,
            done             = done
        )

        # Log trade history
        self.trade_history.append({
            'step':         self.current_step,
            'timestamp':    ts,
            'action':       action,
            'price':        price,
            'inventory':    inventory,
            'realized_pnl': pnl_realized
        })

        return next_obs, reward, done, False, info

    def _can_buy(self):
        return self.fills_ledger.get_inventory() < self.inventory_limit

    def _can_sell(self):
        return self.fills_ledger.get_inventory() > -self.inventory_limit

    def _action_mask(self):
        inv = self.fills_ledger.get_inventory()
        return np.array([
            True,
            inv <  self.inventory_limit,
            inv > -self.inventory_limit
        ], dtype=bool)

    def render(self, mode='human'):
        inv = self.fills_ledger.get_inventory()
        pnl = self.fills_ledger.compute_realized_pnl()
        print(f"Step: {self.current_step}, Inventory: {inv}, Realized PnL: {pnl}")

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def close(self):
        pass
