import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class TradingDaySimulator:
    def __init__(self, log_csv: str, skip: int = 100):
        """
        :param log_csv: Path to training_log.csv
        :param skip: Step sampling rate for animation (every N rows)
        """
        # Load the full DataFrame
        self.df = pd.read_csv(log_csv, parse_dates=["timestamp"])
        self.skip = skip
        # Sample for animation
        self.sampled_df = self.df.iloc[::self.skip].reset_index(drop=True)

        # Precompute static trades for potential static use
        self.trades_static = self.df[self.df["action"].isin([1, 2])].copy()
        self.trades_static["color"] = self.trades_static["action"].map({1: 'green', 2: 'red'})

    def render_static(self, title="Trading Day Overview"):
        # unchanged: use existing Plotly implementation
        raise NotImplementedError("Use existing static implementation")

    def render_video(self, output_path="simulation.mp4"):
        df = self.sampled_df.copy()
        # Parse and drop timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        try:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass

        # Set up the figure: price, action bars, inventory curve, stats
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[4, 1.5, 1.5, 1], hspace=0.4)
        ax_price = fig.add_subplot(gs[0, 0])
        ax_action = fig.add_subplot(gs[1, 0])
        ax_inv_curve = fig.add_subplot(gs[2, 0])
        ax_stats = fig.add_subplot(gs[3, 0])

        # Price plot settings
        ax_price.set_title("Mid Price & Trade Markers")
        ax_price.set_ylabel("Mid Price")
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_price.grid(True, linestyle='--', alpha=0.5)

        # Action counts plot settings
        ax_action.set_title("Action Frequency (Hold/Buy/Sell)")
        ax_action.set_xticks([0, 1, 2])
        ax_action.set_xticklabels(['Hold', 'Buy', 'Sell'])
        ax_action.set_ylabel("Frequency")
        ax_action.set_ylim(0, len(df))  # max possible
        ax_action.grid(axis='y', linestyle='--', alpha=0.5)

        # Inventory curve plot settings
        ax_inv_curve.set_title("Inventory Over Time")
        ax_inv_curve.set_ylabel("Inventory")
        ax_inv_curve.set_xlabel("Step Index")
        ax_inv_curve.grid(True, linestyle='--', alpha=0.5)

        # Stats panel off
        ax_stats.axis('off')

        # Inset for equity
        ax_pnl = inset_axes(ax_price, width="30%", height="30%", loc='upper left', borderpad=2)
        ax_pnl.set_title("Equity Curve", fontsize=8)
        ax_pnl.tick_params(axis='both', which='major', labelsize=6)

        # Initialize price and trades
        price_line, = ax_price.plot([], [], lw=2, color='blue', label='Mid Price')
        trade_scatter = ax_price.scatter([], [], s=40, edgecolor='black', label='Trades')

        # Legend for shading
        long_patch = Patch(color='green', alpha=0.1, label='Long Position')
        short_patch = Patch(color='red', alpha=0.1, label='Short Position')
        handles, labels = ax_price.get_legend_handles_labels()
        handles += [long_patch, short_patch]
        labels += ['Long Position', 'Short Position']
        ax_price.legend(handles, labels, loc='upper right', fontsize=8)

        # Initialize action bar containers
        bar_rects = ax_action.bar([0, 1, 2], [0, 0, 0], color=['gray', 'green', 'red'], align='center')

        # Initialize inventory curve
        inv_line, = ax_inv_curve.plot([], [], lw=1.5, color='purple')

        def init():
            price_line.set_data([], [])
            trade_scatter.set_offsets(np.empty((0, 2)))
            for rect in bar_rects:
                rect.set_height(0)
            inv_line.set_data([], [])
            ax_pnl.clear()
            ax_stats.clear()
            return [price_line, trade_scatter, inv_line] + list(bar_rects)

        def update(frame):
            window = df.iloc[:frame + 1]
            times = window['timestamp']
            prices = window['mid_price']

            # Update price
            price_line.set_data(times, prices)

            # Update trades
            trades = window[window['action'].isin([1, 2])]
            if not trades.empty:
                xs = mdates.date2num(trades['timestamp'])
                ys = trades['mid_price']
                trade_scatter.set_offsets(np.column_stack((xs, ys)))
                trade_scatter.set_facecolors(trades['action'].map({1: 'green', 2: 'red'}).tolist())
            else:
                trade_scatter.set_offsets(np.empty((0, 2)))

            # Shading
            for coll in list(ax_price.collections):
                coll.remove()
            ax_price.fill_between(times, prices.min() - 1, prices.max() + 1,
                                   where=window['inventory'] > 0, color='green', alpha=0.1)
            ax_price.fill_between(times, prices.min() - 1, prices.max() + 1,
                                   where=window['inventory'] < 0, color='red', alpha=0.1)

            # Update equity inset
            ax_pnl.clear()
            ax_pnl.plot(times, window['pnl'], lw=1, color='orange')
            ax_pnl.set_title("Equity Curve", fontsize=8)
            ax_pnl.tick_params(axis='both', labelsize=6)

            # Update action bars
            counts = window['action'].value_counts().reindex([0, 1, 2], fill_value=0)
            for rect, h in zip(bar_rects, counts.values):
                rect.set_height(h)

            # Update inventory curve
            inv_line.set_data(window['step_idx'], window['inventory'])
            ax_inv_curve.set_xlim(0, df['step_idx'].max())
            ax_inv_curve.set_ylim(df['inventory'].min() - 1, df['inventory'].max() + 1)

            # Update stats
            ax_stats.clear()
            stats = window.iloc[-1]
            stats_text = (
                f"Time: {stats['timestamp'].strftime('%H:%M:%S')}\n"
                f"Action: {['Hold','Buy','Sell'][int(stats['action'])]}\n"
                f"Inventory: {int(stats['inventory'])}\n"
                f"PnL: {stats['pnl']:.2f}\n"
                f"Reward: {stats['reward']:.2f}"
            )
            ax_stats.text(0.01, 0.5, stats_text, va='center', ha='left', fontsize=9, family='monospace')
            ax_stats.axis('off')

            # Adjust limits
            ax_price.set_xlim(times.min(), times.max())
            ax_price.set_ylim(prices.min() - 1, prices.max() + 1)

            return [price_line, trade_scatter, inv_line] + list(bar_rects)

        ani = animation.FuncAnimation(fig, update,
                                      frames=len(df), init_func=init,
                                      blit=False, interval=100, repeat=False)
        ani.save(output_path, fps=20, extra_args=['-vcodec', 'libx264'])
        plt.close(fig)

    def render_animated(self, title="Trading Day Simulation"):
        # unchanged: use existing Plotly implementation
        raise NotImplementedError("Use existing animated implementation")
