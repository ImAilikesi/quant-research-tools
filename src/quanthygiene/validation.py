"""Time-series validation splits: walk-forward and purged K-fold with embargo.

Answers: how do I split ordered data without leaking the future into training? Yields integer
index arrays; apply them to any array or DataFrame via ``.iloc``.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning", ch. 7.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def walk_forward_splits(n: int, n_splits: int, test_size: int | None = None,
                        anchored: bool = False, gap: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Successive (train, test) index pairs; each test block follows its training window.

    ``anchored=True`` grows the training set from index 0; otherwise a rolling window of the same
    length as the first training set is used. ``gap`` observations are skipped between train and test.
    """
    test_size = test_size or n // (n_splits + 1)
    first_train = n - n_splits * test_size - gap
    if test_size < 1 or first_train < 1:
        raise ValueError("series too short for this number of splits")
    for k in range(n_splits):
        train_end = first_train + k * test_size
        test_start = train_end + gap
        train_start = 0 if anchored else k * test_size
        yield np.arange(train_start, train_end), np.arange(test_start, test_start + test_size)


def purged_kfold(n: int, n_splits: int = 5, label_horizon: int = 0,
                 embargo: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """K contiguous test folds; training excludes leaked and embargoed observations.

    A label at index i is assumed to use data up to i + ``label_horizon``. Training samples whose
    label window overlaps the test fold are purged; ``embargo`` further observations after the test
    fold are dropped as well.
    """
    if n_splits < 2 or n < n_splits:
        raise ValueError("need 2 <= n_splits <= n")
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    idx = np.arange(n)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        test = idx[a:b]
        leak_before = idx >= a - label_horizon
        keep = ~(leak_before & (idx < b + embargo))
        yield idx[keep], test
