import numpy as np
import pytest

from quanthygiene import sizing, validation


def test_fixed_fractional():
    assert sizing.fixed_fractional(10_000, 0.01, 20, 5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sizing.fixed_fractional(1, 0.01, 0)


def test_kelly():
    assert sizing.kelly_fraction(0.5, 2) == pytest.approx(0.25)
    assert sizing.kelly_fraction(0.3, 1) < 0
    assert sizing.fractional_kelly(0.2, 0.5) == pytest.approx(0.1)
    assert sizing.fractional_kelly(-0.2) == 0
    r = np.random.default_rng(0).normal(0.001, 0.01, 1000)
    assert sizing.kelly_from_returns(r) > 0


def test_vol_target():
    assert sizing.vol_target_exposure(0.1, 0.2) == pytest.approx(0.5)
    assert sizing.vol_target_exposure(0.1, 0.01, max_leverage=3) == 3
    with pytest.raises(ValueError):
        sizing.vol_target_exposure(0.1, 0)
    assert sizing.realized_vol([0.01, -0.01, 0.01, -0.01], 252) > 0


@pytest.mark.parametrize("anchored", [True, False])
def test_walk_forward_no_leak_and_order(anchored):
    splits = list(validation.walk_forward_splits(100, 4, anchored=anchored, gap=2))
    assert len(splits) == 4
    for tr, te in splits:
        assert tr.max() + 2 < te.min()
    assert all(splits[i][1].max() < splits[i + 1][1].min() for i in range(3))
    assert (splits[1][0].min() == 0) == anchored


def test_walk_forward_too_short():
    with pytest.raises(ValueError):
        list(validation.walk_forward_splits(5, 10))


def test_purged_kfold_purges_and_embargoes():
    h, e = 5, 3
    folds = list(validation.purged_kfold(100, 5, h, e))
    assert len(folds) == 5
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert not any(te.min() - h <= i < te.max() + 1 + e for i in tr)
    # all observations are tested exactly once
    assert sorted(np.concatenate([te for _, te in folds])) == list(range(100))


def test_purged_kfold_no_purge_equals_plain_kfold():
    for tr, te in validation.purged_kfold(50, 5):
        assert len(tr) + len(te) == 50


def test_purged_kfold_rejects_bad_args():
    with pytest.raises(ValueError):
        list(validation.purged_kfold(10, 1))
