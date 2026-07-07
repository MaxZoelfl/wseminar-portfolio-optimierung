"""Tests für Purged & Embargoed Cross-Validation (López de Prado 2018).

FÜR EINSTEIGER: Automatische Prüfprogramme (Erklärung des Testprinzips im
Kopf von tests/test_metrics.py). Diese Datei prüft die leckagefreie
Datenaufteilung aus portfolio/cross_validation.py: Werden Trainings- und
Testzeiträume wirklich sauber getrennt, und werden die Sicherheitsabstände
(Purge davor, Embargo danach) tatsächlich freigehalten? Statt echter Daten
genügen künstliche Zeitstempel (Periodennummern 0, 1, 2, …).
"""
import numpy as np
import pytest

from portfolio.cross_validation import purged_kfold_splits


def _panel_times(n_periods, n_assets):
    # n_assets Zeilen je Periode (Panel)
    # Beispiel (3 Perioden, 2 Assets): [0, 0, 1, 1, 2, 2] — wie im echten
    # Datensatz, wo jeder Monat 15 Zeilen (eine je Aktie) hat.
    return np.repeat(np.arange(n_periods), n_assets)


def test_returns_requested_number_of_folds():
    # Wer 4 Aufteilungen bestellt, muss 4 bekommen.
    splits = purged_kfold_splits(_panel_times(20, 3), n_splits=4, embargo_pct=0.1)
    assert len(splits) == 4


def test_train_test_disjoint_and_panel_grouped():
    # Kernregel jeder Kreuzvalidierung: keine Zeile darf gleichzeitig im
    # Training UND im Test sein — und hier zusätzlich: auch keine PERIODE
    # (sonst stünde Aktie A vom März im Training und Aktie B vom März im Test).
    times = _panel_times(20, 3)
    for train_idx, test_idx in purged_kfold_splits(times, n_splits=4, embargo_pct=0.1):
        assert set(train_idx).isdisjoint(test_idx)        # disjunkt
        tr_periods = set(times[train_idx])
        te_periods = set(times[test_idx])
        # keine Periode gleichzeitig in Train und Test (Panel sauber getrennt)
        assert tr_periods.isdisjoint(te_periods)


def test_purge_and_embargo_remove_adjacent_periods():
    # Prüft die beiden Sicherheitsabstände einzeln nach: Die Periode direkt
    # VOR dem Testfenster (Purge) und die Perioden direkt DANACH (Embargo)
    # dürfen nicht im Training auftauchen. Die "oder"-Zusätze fangen die
    # Ränder ab (vor Periode 0 bzw. nach der letzten gibt es nichts).
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
    # Plausibilitätsprüfung: Ein Embargo SPERRT zusätzliche Monate, also muss
    # mit Embargo insgesamt weniger Trainingsmaterial übrig bleiben als ohne.
    times = _panel_times(30, 2)
    n_no = sum(len(tr) for tr, _ in purged_kfold_splits(times, 5, embargo_pct=0.0))
    n_emb = sum(len(tr) for tr, _ in purged_kfold_splits(times, 5, embargo_pct=0.2))
    assert n_emb < n_no            # Embargo entfernt zusätzliche Trainingszeilen


def test_raises_with_too_few_periods():
    # Bei absurd wenigen Perioden (3 Monate, aber 5 Splits gewünscht) soll die
    # Funktion mit einer klaren Fehlermeldung abbrechen. pytest.raises prüft
    # genau das: Der Test ist GRÜN, wenn der erwartete Fehler auftritt.
    with pytest.raises(ValueError):
        purged_kfold_splits(_panel_times(3, 2), n_splits=5)
