import numpy as np
import pandas as pd
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import ttest_1samp

def cluster_strategy_rediscovery(df, hidden_array, pca_components=3, min_samples=20, min_cluster_size=50):
    """
    PCA + HDBSCAN clustering for strategy rediscovery

    PCA to reduce feature dimensionality
    HDBSCAN to cluster the data - known strategies tend to form dense clusters based on timing, imbalance, inventory

    features used: step_idx, inventory, spread, volume_imbalance
    1. step_idx: allows clusters to distinguish between fast and slow strategies
    2. inventory: can distinguish between strategies such as market making vs. momentum trading
    3. spread and volume_imbalance: can distinguish between spread-capture and directional strategies

    features were engineered to potentially distinguish between strategies. some of these include
    1. price action correlation: price_velocity * action (1 for buy, -1 for sell, 0 otherwise)
    2. directional correctness: based on if action is "correct" based on price_velocity

    returns array of cluster labels
    """

    df = df.copy()

    if 'episode_id' not in df.columns:
        df['episode_id'] = 0
      
    df['action'] = pd.to_numeric(df['action'], errors='coerce').fillna(0).astype(int)

    # normalized timestamp from step_idx
    df["step_idx_norm"] = df["step_idx"] / df["step_idx"].max()

    # price action correlation - captures differences between trading strategies based on market movement and action
    df["px_act_corr"] = df["price_velocity"] * np.where(df["action"]==1,  1, np.where(df.action==2, -1, 0))

    # directional correctness - captures frequency of correct vs incorrect actions
    df["dir_correct"] = ((df["price_velocity"]>0)&(df["action"]==1)) | ((df["price_velocity"]<0)&(df["action"]==2))
    df["dir_correct"] = df["dir_correct"].astype(int)

    # can distinguish between strategies that trade continuously vs. those that trade more infrequently
    if 'episode_id' in df.columns and df['episode_id'].nunique() > 1:
        df['trade_count'] = df.groupby('episode_id')['action'].cumsum()
    else:
        df['trade_count'] = df['action'].cumsum()
    df["churn"] = df["trade_count"].diff().fillna(0)

    # smooths out price velocity to reduce noise
    df["vel_ma5"] = df["price_velocity"].rolling(5).mean().fillna(0)

    # how fast and which direction the agent's position is moving
    df["inv_diff"] = df["inventory"].diff().fillna(0)

    feat_cols = [
        "step_idx_norm", "inventory", "spread", "volume_imbalance",
        "price_velocity", "px_act_corr", "dir_correct", "churn",
        "vel_ma5", "inv_diff", "reward", "pnl"
    ]

    # feat_cols = [
    #     "step_idx_norm", "inventory", "spread", "volume_imbalance", "price_velocity"
    # ]

    X = StandardScaler().fit_transform(df[feat_cols].fillna(0))
    pca = PCA(n_components=pca_components, random_state=0)

    if hidden_array.shape[0] != X.shape[0]:
        raise ValueError("hidden_array must have same number of rows as df")
    # Optionally reduce hidden dim with PCA to avoid extremely large dims
    hidden_pca = PCA(n_components=min(pca_components, hidden_array.shape[1]), random_state=0)
    H = hidden_pca.fit_transform(hidden_array)

    X_combined = np.hstack([X, H])

    pca = PCA(n_components=pca_components, random_state=0)
    Xr = pca.fit_transform(X_combined)

    # clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    # clusterer = KMeans(n_clusters=2, random_state=0)
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(Xr)
    return labels

def cluster_novel_strategy(df, hidden_array, tsne_components=2, min_samples=10, min_cluster_size=50):
    """
    t-SNE + DBSCAN clustering for novel strategy discovery

    t-SNE good at preserving local neighborhoods of high-dimensional data 
    (exaggerates small, tight clusters that might correspond to a novel strategy)
    DBSCAN to cluster these novel strategy clusters

    features used: reward, spread, volume_imbalance
    1. reward: emphasizes trades that have unusually high profitability and risk-adjusted returns
    2. spread and volume_imbalance: helps explain how the reward was achieved

    returns array of cluster labels
    """

    feat_cols = ['reward', 'spread', 'volume_imbalance']
    X_base = StandardScaler().fit_transform(df[feat_cols].fillna(0).values)

    hidden_pca = PCA(n_components=min(tsne_components, hidden_array.shape[1]), random_state=0)
    H = hidden_pca.fit_transform(hidden_array)

    X_combined = np.hstack([X_base, H])

    # t-SNE embedding
    tsne = TSNE(n_components=tsne_components, init='random', random_state=0)
    Xr = tsne.fit_transform(X_combined)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(Xr)

    return labels

def evaluate_risk_awareness(df, drawdown_q=0.95, inv_std_q=0.95, lag_q=0.95, iforest_contam=0.05):
    """
    uses set thresholds and isolation forest to identify risk-aware strategies

    features used: max_drawdown, inventory_std, reactivity_lag
    1. max_drawdown: size of worst loss in signficant when measuring risk
    2. inventory_std: high std indicates rapid position changes -> high risk
    3. reactivity_lag: high lag indicates slow reaction to market changes and vice versa

    risk categories are "safe", "risky"

    returns DataFrame with index = episode_id and columns = [max_drawdown, inventory_std, reactivity_lag, risk_category]
    """

    groups = df.groupby("episode_id")
    data = []

    for ep, g in groups:
        pnl = g['pnl'].values

        dd = np.maximum.accumulate(pnl) - pnl
        max_dd = np.max(dd) # max drawdown - how bad was the worst loss
        
        inv = g['inventory'].values
        inv_std = np.std(inv) # inv std tells how much agent changes position - higher std = higher risk

        lags = []

        for i in range(1, len(pnl)):
            if pnl[i] < pnl[i-1]:
                # we measure lag when drawdown occurs
                base = abs(inv[i-1]) # check position size at time of drawdown

                for j in range(i+1, len(inv)):
                    if inv[j] < base:
                        lag = g["step_idx"].iloc[j] - g["step_idx"].iloc[i] # reactivity lag
                        lags.append(lag)
                        break
            
        react_lag = np.mean(lags) if len(lags) > 0 else 0 # avg reactivity lag for episode
        data.append((ep, max_dd, inv_std, react_lag))
    
    risk_df = pd.DataFrame(data, columns=["episode_id", "max_dd", "inv_std", "react_lag"]).set_index("episode_id")

    thresh_dd = risk_df["max_dd"].quantile(drawdown_q)
    thresh_std = risk_df["inv_std"].quantile(inv_std_q)
    thresh_lag = risk_df["react_lag"].quantile(lag_q)

    def categorize(row):
        if row["max_dd"] < thresh_dd and row["inv_std"] < thresh_std and row["react_lag"] < thresh_lag:
            return "safe"
        else:
            return "risky"
    
    risk_df["risk_category"] = risk_df.apply(categorize, axis=1)

    iso_forest = IsolationForest(contamination=iforest_contam, random_state=0)
    scores = iso_forest.fit_predict(risk_df[["max_dd", "inv_std", "react_lag"]])

    risk_df["anomaly"] = (scores == -1)
    
    return risk_df

def evaluate_profitability(df):
    """
    assigns episodes to profit categories (low/mid/high) based on profit quantiles
    performs one-sample t-test to check if profit is significantly different from zero

    features used: end_pnl, avg_reward, trade_freq
    1. end_pnl: total profit/loss at end of episode, direct measure of agent performance
    2. avg_reward: average reward per step, normalized measure of performance
    3. trade_freq: number of non-hold trades per episode, distinguishes between few big trades and many small trades

    returns DataFrame with index = episode_id and columns = end_pnl, avg_reward, trade_freq, profit_quantile
    returns t-statistic and p-value from one-sample t-test
    """

    group = df.groupby("episode_id")
    end_pnl = group["pnl"].last().rename("end_pnl")
    avg_reward = group["reward"].mean().rename("avg_reward")
    trade_freq = group["action"].apply(lambda x: np.sum(x != 0)).rename("trade_freq")

    profit_df = pd.concat([end_pnl, avg_reward, trade_freq], axis=1)

    try:
        profit_df["profit_quantile"] = pd.qcut(profit_df["end_pnl"], q=2, labels=['low', 'high'], duplicates='drop')
    except ValueError:
        profit_df["profit_quantile"] = 'low'

    t_stat, p_val = ttest_1samp(profit_df["end_pnl"], popmean=0.0)

    return profit_df, t_stat, p_val
