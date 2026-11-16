"""Correlation clustering for a set of strategies or assets.

Answers: which return streams are really the same bet? Uses the distance
``d = sqrt(0.5 * (1 - corr))`` and average-linkage agglomerative clustering (pure numpy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_distance(returns: pd.DataFrame) -> pd.DataFrame:
    """Distance matrix sqrt(0.5 * (1 - corr)); 0 = identical, 1 = perfectly anti-correlated."""
    corr = returns.corr()
    return np.sqrt(0.5 * (1 - corr.clip(-1, 1)))


def cluster(returns: pd.DataFrame, n_clusters: int = 3) -> dict:
    """Average-linkage clustering into ``n_clusters``; returns labels, groups and merge order."""
    d = correlation_distance(returns)
    names = list(d.columns)
    if not 1 <= n_clusters <= len(names):
        raise ValueError("n_clusters must be between 1 and the number of columns")
    groups: dict[int, list[int]] = {i: [i] for i in range(len(names))}
    dist = d.to_numpy()
    merges = []
    while len(groups) > n_clusters:
        keys = list(groups)
        best = (np.inf, None, None)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                dab = dist[np.ix_(groups[keys[a]], groups[keys[b]])].mean()
                if dab < best[0]:
                    best = (dab, keys[a], keys[b])
        dab, ka, kb = best
        merges.append({"distance": float(dab), "joined": [[names[i] for i in groups[ka]],
                                                           [names[i] for i in groups[kb]]]})
        groups[ka] = groups[ka] + groups.pop(kb)
    result = [[names[i] for i in g] for g in groups.values()]
    labels = {names[i]: k for k, g in enumerate(groups.values()) for i in g}
    return {"clusters": result, "labels": labels, "merges": merges,
            "mean_pairwise_corr": float(returns.corr().to_numpy()[np.triu_indices(len(names), 1)].mean())}
