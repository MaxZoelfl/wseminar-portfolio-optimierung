"""
Purged & Embargoed Cross-Validation für Finanz-Zeitreihen (Panel-Daten).

Standard-k-Fold und auch TimeSeriesSplit leaken, wenn die Labels über mehrere
Perioden überlappen — hier: das monatliche Target wird aus überlappenden
Tagesfenstern gebildet (Momentum 252d, rollierendes Alpha/Beta), und das Label
eines Monats t reicht bis t+1. Dadurch teilen benachbarte Train-/Test-Monate
Information.

Abhilfe (López de Prado 2018, Kap. 7):
  - PURGING: entferne aus dem Training alle Beobachtungen, deren Label-Fenster
    mit dem Test-Fenster überlappt.
  - EMBARGO: entferne zusätzlich Beobachtungen unmittelbar NACH dem Test-Fenster
    (Schutz gegen serielle Korrelation).

Diese Implementierung ist PANEL-fähig: pro Periode (Monatsende) existieren
mehrere Zeilen (ein Asset je Zeile); gesplittet wird nach Perioden, sodass nie
ein Teil der Assets einer Periode im Training und ein anderer im Test landet.

Referenz: López de Prado, M. (2018): Advances in Financial Machine Learning,
Kap. 7 (Cross-Validation in Finance). Wiley.

Layer: nutzt nur numpy/pandas, keine projektinternen Module.
"""
import numpy as np
import pandas as pd


def purged_kfold_splits(times, n_splits=5, embargo_pct=0.0, purge=1):
    """
    Erzeugt purged & embargoed Train/Test-Splits für Panel-Daten.

    Parameter
    ---------
    times       : array-artig, Periode je Zeile (z. B. Monatsende-Zeitstempel).
                  Länge = Anzahl Beobachtungen; Wiederholungen (mehrere Assets je
                  Periode) sind erlaubt.
    n_splits    : Anzahl zeitlich zusammenhängender Test-Folds.
    embargo_pct : Embargo-Länge als Anteil der Anzahl eindeutiger Perioden.
    purge       : Anzahl überlappender Perioden, die rund um das Test-Fenster aus
                  dem Training entfernt werden (Label-Überlappung; hier 1, da das
                  Monatslabel eine Folgeperiode umfasst).

    Rückgabe
    --------
    Liste von (train_idx, test_idx) Positions-Arrays — direkt als sklearn-`cv`
    nutzbar.
    """
    t = pd.Index(times)
    periods = t.unique().sort_values()
    P = len(periods)
    if P < n_splits + purge + 1:
        raise ValueError(
            f"Zu wenige Perioden ({P}) für purged CV mit n_splits={n_splits}, "
            f"purge={purge}."
        )
    pos = pd.Series(np.arange(P), index=periods)
    row_pos = pos.reindex(t).to_numpy()                 # Perioden-Index je Zeile
    embargo = int(np.ceil(embargo_pct * P))

    bounds = np.linspace(0, P, n_splits + 1).astype(int)
    splits = []
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1] - 1           # Test-Perioden [lo, hi]
        if hi < lo:
            continue
        test_mask = (row_pos >= lo) & (row_pos <= hi)
        # Ausschlusszone fürs Training: Purge davor, Embargo danach
        excl_lo, excl_hi = lo - purge, hi + embargo
        train_mask = (row_pos < excl_lo) | (row_pos > excl_hi)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        if len(test_idx) > 0 and len(train_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits
