import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from evaluation.cluster import (
    cluster_strategy_rediscovery,
    cluster_novel_strategy,
    evaluate_risk_awareness,
    evaluate_profitability
)

def plot_reward_per_episode(log_df):
    """
    plot the reward per episode
    """
    df = log_df.copy()
    if 'episode_id' not in df.columns:
        df['episode_id'] = 0
    totals = df.groupby('episode_id')['reward'].sum()
    plt.figure()
    plt.plot(totals.index, totals.values)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid()
    plt.show()

def plot_pnl_over_time(log_df):
    """
    plot the pnl over time
    """
    df = log_df.copy()
    if 'episode_id' not in df.columns:
        df['episode_id'] = 0
    plt.figure()
    groups = log_df.groupby('episode_id')
    for ep, g in groups:
        plt.plot(g['step_idx'], g['pnl'], label=f'Episode {ep}')
    plt.xlabel('Step Index')
    plt.ylabel('PnL')
    plt.title('PnL Over Time')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

def plot_action_frequency(log_df):
    """
    plot the action frequency

    0 - hold
    1 - buy
    2 - sell
    3 - quote
    4 - cancel
    """
    action_map = {
        0: 'hold',
        1: 'buy',
        2: 'sell',
        3: 'quote',
        4: 'cancel'
    }
    df = log_df.copy()
    df['action'] = pd.to_numeric(df['action'], errors='coerce').fillna(0).astype(int)

    action_freq = df['action'].value_counts().sort_index()
    all_actions = list(action_map.keys())
    counts = [action_freq.get(a, 0) for a in all_actions]
    tick_labels = [action_map[a] for a in all_actions]

    plt.figure()
    plt.bar(all_actions, counts, tick_label=tick_labels)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Action Frequency Distribution')
    plt.show()

def plot_cluster_scatter(
        log_df,
        hidden_array,
        method,
        pca_components=3,
        tsne_components=2,
        min_samples=20, 
        min_cluster_size=50, 
        **kwargs):
    """
    plot the cluster scatter plot
    method: 'rediscovery' or 'novel'
    """
    if method == "rediscovery":
        feat_cols = [
            "step_idx",
            "inventory",
            "spread",
            "volume_imbalance",
            "price_velocity",
            "reward",
            "pnl",
        ]
        X_core = log_df[feat_cols].fillna(0).values
        X_core = StandardScaler().fit_transform(X_core)

        hidden_pca = PCA(
            n_components=min(pca_components, hidden_array.shape[1]),
            random_state=0
        )
        H = hidden_pca.fit_transform(hidden_array)

        X_combined = np.hstack([X_core, H])
        emb = PCA(n_components=2, random_state=0).fit_transform(X_combined)

        labels = cluster_strategy_rediscovery(
            log_df,
            hidden_array,
            pca_components=pca_components,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )

    else:
        feat_cols = ["reward", "spread", "volume_imbalance"]
        X_base = log_df[feat_cols].fillna(0).values
        X_base = StandardScaler().fit_transform(X_base)

        hidden_pca = PCA(
            n_components=min(tsne_components, hidden_array.shape[1]),
            random_state=0
        )
        H = hidden_pca.fit_transform(hidden_array)

        X_combined = np.hstack([X_base, H])
        emb = TSNE(n_components=2, random_state=0).fit_transform(X_combined)

        labels = cluster_novel_strategy(
            log_df,
            hidden_array,
            tsne_components=tsne_components,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )

    emb_df = pd.DataFrame(emb, columns=["emb1", "emb2"])
    emb_df["cluster"] = labels

    plt.figure(figsize=(6, 5))
    for cluster in sorted(emb_df["cluster"].unique()):
        subset = emb_df[emb_df["cluster"] == cluster]
        name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        plt.scatter(
            subset["emb1"],
            subset["emb2"],
            label=name,
            alpha=0.6,
            s=20
        )

    plt.xlabel("emb1")
    plt.ylabel("emb2")
    plt.title(f"Cluster Scatter ({method})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_inventory(log_df):
    """
    plot the inventory over time
    """
    df = log_df.copy()
    if 'episode_id' not in df.columns:
        df['episode_id'] = 0
    plt.figure()
    for ep, g in log_df.groupby('episode_id'):
        plt.plot(g['step_idx'], g['inventory'], label=f'Episode {ep}')
    plt.xlabel('Step Index')
    plt.ylabel('Inventory')
    plt.title('Inventory Over Time')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

def plot_risk_scatter(log_df):
    """
    scatter plot of inventory volatility vs. max drawdown, colored by risk category
    """
    risk_df = evaluate_risk_awareness(log_df)
    plt.figure()
    for cat in risk_df['risk_category'].unique():
        cat_data = risk_df[risk_df['risk_category'] == cat]
        plt.scatter(cat_data['inv_std'], cat_data['max_dd'], label=cat, alpha=0.5)
    plt.xlabel('Inventory Std')
    plt.ylabel('Max Drawdown')
    plt.title('Risk Scatter Plot')
    plt.grid()
    plt.legend()
    plt.show()

def plot_anomaly(log_df):
    """
    histogram of anomaly flags
    """
    risk_df = evaluate_risk_awareness(log_df)
    counts = risk_df['anomaly'].value_counts()
    plt.figure()
    plt.bar(['Normal', 'Anomaly'], [counts.get(False, 0), counts.get(True, 0)])
    plt.xlabel('Anomaly Flag')
    plt.ylabel('Count')
    plt.title('Anomaly Detection Plot')
    plt.show()

def plot_profitability(log_df):
    """
    plot boxplot of end-of-episode pnl by profit quantile

    prints t-statistic and p-value of one-sample t-test
    - t-statistic: how many standard errors the sample mean is from 0
    - p-value: probability of observing a t-statistic as extreme as the one calculated
    """
    profit_df, t_stat, p_val = evaluate_profitability(log_df)
    print(f"Mean end-Pnl vs 0: t={t_stat:.3f}, p={p_val:.3f}")
    plt.figure()
    profit_df.boxplot(column='end_pnl', by='profit_quantile')
    plt.xlabel('Profit Quantile')
    plt.ylabel('End PnL')
    plt.title('Profitability by Quantile')
    plt.suptitle('')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load the log file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(BASE_DIR, 'mock_training_log.csv')
    log_df = pd.read_csv(log_file)
    # plot_reward_per_episode(log_df)
    # plot_pnl_over_time(log_df)
    # plot_action_frequency(log_df)

    # labels = cluster_strategy_rediscovery(log_df)

    # plot_inventory(log_df)

    # plot_profitability(log_df)