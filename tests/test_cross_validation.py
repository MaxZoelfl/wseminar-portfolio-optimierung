"""Tests für Purged & Embargoed Cross-Validation (López de Prado 2018)."""
import numpy as np
import pytest

from portfolio.cross_validation import purged_kfold_splits


def _panel_times(n_periods, n_assets):
    # n_assets Zeilen je Periode (Panel)
    return np.repeat(np.arange(n_periods), n_assets)


def test_returns_requested_number_of_folds():
    splits = purged_kfold_splits(_panel_times(20, 3), n_splits=4, embargo_pct=0.1)
    assert len(splits) == 4


def test_train_test_disjoint_and_panel_grouped():
    times = _panel_times(20, 3)
    for train_idx, test_idx in purged_kfold_splits(times, n_splits=4, embargo_pct=0.1):
        assert set(train_idx).isdisjoint(test_idx)        # disjunkt
        tr_periods = set(times[train_idx])
        te_periods = set(times[test_idx])
        # keine Periode gleichzeitig in Train und Test (Panel sauber getrennt)
        assert tr_periods.isdisjoint(te_periods)


def test_purge_and_embargo_remove_adjacent_periods():
    times = _panel_times(20, 3)
    embargo_periods = int(np.ceil(0.10 * 20))   # = 2
    for train_idx, test_idx in purged_kfold_splits(times, n_splits=4,
                                                    embargo_pct=0.10, purge=1):
        tr_periods = set(times[train_idx])
        lo, hi = times[test_idx].min(), times[test_idx].max()
        # Purge: die Periode unmittelbar vor dem Test fehlt im Training
        assert (lo - 1) not in tr_periods or lo == 0
        # Embargo: die Perioden unmittelbar nach dem Test fehlen im Training
        for e in range(1, embargo_periods + 1):
            assert (hi + e) not in tr_periods or (hi + e) >= 20


def test_no_embargo_keeps_more_training_than_with_embargo():
    times = _panel_times(30, 2)
    n_no = sum(len(tr) for tr, _ in purged_kfold_splits(times, 5, embargo_pct=0.0))
    n_emb = sum(len(tr) for tr, _ in purged_kfold_splits(times, 5, embargo_pct=0.2))
    assert n_emb < n_no            # Embargo entfernt zusätzliche Trainingszeilen


def test_raises_with_too_few_periods():
    with pytest.raises(ValueError):
        purged_kfold_splits(_panel_times(3, 2), n_splits=5)
