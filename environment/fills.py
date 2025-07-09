# File: agitrader/environment/fills.py

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Fill:
    timestamp: str
    price: float
    size: int
    side: str  # 'buy' or 'sell'
    status: str = "filled"  # Future use: "pending", "partial", etc.

class FillsLedger:
    def __init__(self):
        self.fills: List[Fill] = []

    def add_fill(self, timestamp: str, price: float, size: int, side: str):
        assert side in ('buy', 'sell'), "Side must be 'buy' or 'sell'"
        self.fills.append(Fill(timestamp, price, size, side))

    def reset(self):
        self.fills.clear()

    def get_inventory(self):
        buys = sum(f.size for f in self.fills if f.side == 'buy')
        sells = sum(f.size for f in self.fills if f.side == 'sell')
        return buys - sells  # positive: net long, negative: net short

    def get_gross_inventory(self):
        return sum(f.size for f in self.fills)

    def compute_realized_pnl(self):
        """
        FIFO-based realized PnL calculation.
        """
        buys = []
        realized = 0.0

        for fill in self.fills:
            if fill.side == 'buy':
                buys.append([fill.price, fill.size])
            elif fill.side == 'sell':
                qty_to_match = fill.size
                while qty_to_match > 0 and buys:
                    entry_price, entry_qty = buys[0]
                    matched_qty = min(qty_to_match, entry_qty)
                    realized += matched_qty * (fill.price - entry_price)
                    qty_to_match -= matched_qty
                    if matched_qty == entry_qty:
                        buys.pop(0)
                    else:
                        buys[0][1] -= matched_qty
        return realized

    def compute_unrealized_pnl(self, current_price: float):
        """
        Unrealized PnL based on remaining open buys/sells.
        Assumes long-only exposure for now.
        """
        net_pos = self.get_inventory()
        if net_pos == 0:
            return 0.0

        # Accumulate cost basis of unmatched fills
        open_value = 0.0
        qty_left = abs(net_pos)
        direction = 'buy' if net_pos > 0 else 'sell'
        matched_qty = 0

        if direction == 'buy':
            for fill in self.fills:
                if fill.side == 'buy':
                    if qty_left == 0:
                        break
                    qty = min(fill.size, qty_left)
                    open_value += qty * fill.price
                    qty_left -= qty
            return net_pos * (current_price - open_value / abs(net_pos))
        else:
            for fill in self.fills:
                if fill.side == 'sell':
                    if qty_left == 0:
                        break
                    qty = min(fill.size, qty_left)
                    open_value += qty * fill.price
                    qty_left -= qty
            return abs(net_pos) * (open_value / abs(net_pos) - current_price)

    @property
    def net_position(self) -> int:
        """
        Current net inventory.
        """
        return sum(fill.size if fill.side == "buy" else -fill.size for fill in self.fills)

    @property
    def realized_pnl(self) -> float:
        """
        Realized profit and loss from all closed trades.
        """
        inventory = []
        pnl = 0.0

        for fill in self.fills:
            if fill.side == "buy":
                inventory.append(fill)
            else:  # sell
                size_to_match = fill.size
                while size_to_match > 0 and inventory:
                    buy_fill = inventory.pop(0)
                    matched_size = min(buy_fill.size, size_to_match)
                    pnl += matched_size * (fill.price - buy_fill.price)

                    # If not fully consumed, push remainder back
                    if matched_size < buy_fill.size:
                        inventory.insert(0, Fill(
                            timestamp=buy_fill.timestamp,
                            price=buy_fill.price,
                            size=buy_fill.size - matched_size,
                            side="buy"
                        ))

                    size_to_match -= matched_size

        return pnl

