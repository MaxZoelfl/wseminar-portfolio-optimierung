"""
Kontrollrechnung zu § 2.1 — jede Zahl des Abschnitts von Hand nachvollziehbar.

WOZU DIESE DATEI?
Sie rechnet die Zahlen aus § 2.1 der Seminararbeit Schritt für Schritt noch
einmal nach und zeigt dabei die Zwischenergebnisse. Damit lässt sich jede
Angabe im Text überprüfen, ohne den vollständigen Backtest laufen zu lassen
(der braucht knapp 30 Minuten, diese Datei zwei Sekunden).

Sie ändert NICHTS am Projekt und schreibt keine Datei. Sie liest nur die
eingefrorenen Kursdaten `data/prices.pkl` und gibt Zahlen aus.

AUFRUF (aus dem Ordner Code/):
    venv/bin/python nachrechnen_kapitel2.py

AUFBAU — die fünf Schritte entsprechen dem Aufbau von § 2.1:
    Schritt 1  Formel (2.1): aus Kursen werden Renditen
    Schritt 2  Formel (2.2): Erwartungswert und Varianz je Titel
    Schritt 3  Formel (2.2): Kovarianz und Korrelation zwischen zwei Titeln
    Schritt 4  Formel (2.3): vom Titel zum Portfolio
    Schritt 5  Gegenprobe: Formel gegen direkte Berechnung
"""

import numpy as np
import pandas as pd

# Die 15 Titel des Anlageuniversums, in der Reihenfolge aus config.py.
TICKERS = ["AAPL", "MSFT", "NVDA", "JNJ", "UNH", "JPM", "GS", "PG",
           "KO", "XOM", "CAT", "HON", "VZ", "PLD", "LIN"]

# Ein Börsenjahr hat rund 252 Handelstage. Mit dieser Zahl wird von
# Tages- auf Jahreswerte umgerechnet ("annualisiert").
HANDELSTAGE = 252


def trennlinie(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


# ────────────────────────────────────────────────────────────────────────
# Daten laden
# ────────────────────────────────────────────────────────────────────────
# prices.pkl enthält die am 15.08.2026 abgerufenen Schlusskurse, bereinigt
# um Splits und Dividenden. Die Datei ist eingefroren, damit die Rechnung
# reproduzierbar bleibt — Yahoo liefert bei jedem neuen Abruf leicht andere
# Werte (siehe Anhang C der Arbeit).
kurse = pd.read_pickle("data/prices.pkl")[TICKERS]

# Zeitraum: genau der des Backtests. Die Datei daily_returns.csv aus dem
# Ergebnisordner legt Anfangs- und Enddatum fest, damit hier dieselbe
# Stichprobe verwendet wird wie in Kapitel 6.
backtest = pd.read_csv("output/daily_returns.csv", index_col=0, parse_dates=True)
start, ende = backtest.index[0], backtest.index[-1]


# ────────────────────────────────────────────────────────────────────────
# Schritt 1 — Formel (2.1): aus Kursen werden Renditen
# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 1 — Formel (2.1): R = (P_t - P_{t-1}) / P_{t-1}")

# .pct_change() rechnet genau diese Formel für jeden Tag und jeden Titel aus.
# .dropna() wirft die erste Zeile weg — für sie gibt es keinen Vortageskurs.
renditen = kurse.pct_change().dropna().loc[start:ende]

# Ein einzelnes Beispiel zum Nachrechnen auf dem Taschenrechner:
titel, tag = "KO", renditen.index[0]
vortag = kurse.index[kurse.index.get_loc(tag) - 1]
p_alt, p_neu = kurse.loc[vortag, titel], kurse.loc[tag, titel]
print(f"  Beispiel {titel} am {tag.date()}:")
print(f"    Kurs am Vortag  P_(t-1) = {p_alt:.6f}")
print(f"    Kurs am Tag     P_t     = {p_neu:.6f}")
print(f"    Rendite R = ({p_neu:.6f} - {p_alt:.6f}) / {p_alt:.6f} = {(p_neu - p_alt) / p_alt:.8f}")
print(f"    aus der Tabelle:                                       {renditen.loc[tag, titel]:.8f}")
print(f"\n  Ergebnis: {len(renditen)} Handelstage x {len(TICKERS)} Titel")
print(f"            vom {renditen.index[0].date()} bis {renditen.index[-1].date()}")


# ────────────────────────────────────────────────────────────────────────
# Schritt 2 — Formel (2.2): Erwartungswert und Varianz je Titel
# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 2 — Formel (2.2): mu_i und sigma_i je Titel")

# Der Erwartungswert E[R] ist theoretisch. Aus Daten schätzt man ihn durch
# den Mittelwert der beobachteten Renditen. Dieser Unterschied — gesuchte
# Größe gegen geschätzte Größe — ist das Thema von § 2.5.
mu_taeglich = renditen.mean()
var_taeglich = renditen.var()            # pandas teilt durch (T-1), nicht T

# Annualisieren: Der Mittelwert wird mit 252 multipliziert (252 Tage im
# Jahr), die Varianz ebenfalls mit 252. Die Standardabweichung dagegen mit
# der WURZEL aus 252, weil sie die Wurzel der Varianz ist.
mu = mu_taeglich * HANDELSTAGE
sigma = np.sqrt(var_taeglich * HANDELSTAGE)

tabelle = pd.DataFrame({
    "mu (% p.a.)": mu * 100,
    "sigma (% p.a.)": sigma * 100,
}).sort_values("sigma (% p.a.)")
print(tabelle.round(2).to_string())

print(f"\n  Nachrechnen am Beispiel KO:")
print(f"    Mittelwert der Tagesrenditen  = {mu_taeglich['KO']:.8f}")
print(f"    mal 252                       = {mu['KO']:.6f}  = {mu['KO'] * 100:.2f} % p. a.")
print(f"    Varianz der Tagesrenditen     = {var_taeglich['KO']:.10f}")
print(f"    mal 252, dann Wurzel          = {sigma['KO']:.6f}  = {sigma['KO'] * 100:.2f} % p. a.")

print(f"\n  ---> für § 2.1: Spanne der Renditen        "
      f"{mu.min() * 100:.1f} % ({mu.idxmin()})  bis  {mu.max() * 100:.1f} % ({mu.idxmax()})")
print(f"  ---> für § 2.1: Spanne der Volatilitaeten  "
      f"{sigma.min() * 100:.1f} % ({sigma.idxmin()})  bis  {sigma.max() * 100:.1f} % ({sigma.idxmax()})")


# ────────────────────────────────────────────────────────────────────────
# Schritt 3 — Formel (2.2): Kovarianz und Korrelation
# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 3 — Formel (2.2): sigma_ij = rho_ij * sigma_i * sigma_j")

Sigma = renditen.cov() * HANDELSTAGE     # annualisierte Kovarianzmatrix
Korr = renditen.corr()                   # Korrelationen (dimensionslos)

a, b = "KO", "CAT"
print(f"  Beispielpaar {a} und {b}:")
print(f"    sigma_{a}   = {sigma[a]:.6f}")
print(f"    sigma_{b}  = {sigma[b]:.6f}")
print(f"    rho        = {Korr.loc[a, b]:.6f}")
print(f"    Kovarianz aus der Matrix          sigma_ij = {Sigma.loc[a, b]:.8f}")
print(f"    Probe rho * sigma_i * sigma_j            = "
      f"{Korr.loc[a, b] * sigma[a] * sigma[b]:.8f}   <- muss gleich sein")

print(f"\n  Die Matrix Sigma ist {Sigma.shape[0]} x {Sigma.shape[1]} = {Sigma.size} Eintraege.")
print(f"  Davon {len(TICKERS)} Varianzen auf der Diagonale und "
      f"{len(TICKERS) * (len(TICKERS) - 1)} Kovarianzen daneben.")
print(f"  Weil sigma_ij = sigma_ji gilt, sind darunter nur "
      f"{len(TICKERS) * (len(TICKERS) - 1) // 2} verschiedene Werte.")


# ────────────────────────────────────────────────────────────────────────
# Schritt 4 — Formel (2.3): vom Titel zum Portfolio
# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 4 — Formel (2.3): mu_P = w' mu  und  sigma_P^2 = w' Sigma w")

N = len(TICKERS)
w = np.ones(N) / N                       # Gleichgewichtung: jeder Titel 1/15

mu_P = w @ mu.values                     # das Skalarprodukt w' mu
var_P = w @ Sigma.values @ w             # die Doppelsumme w' Sigma w
sigma_P = np.sqrt(var_P)

print(f"  Gewichtsvektor w = (1/{N}, ..., 1/{N}),  Summe = {w.sum():.4f}")
print(f"\n  mu_P     = w' mu      = {mu_P:.6f}  = {mu_P * 100:.2f} % p. a.")
print(f"  Probe: einfacher Durchschnitt der 15 Einzelrenditen "
      f"= {mu.mean() * 100:.2f} %   <- identisch")
print(f"\n  sigma_P^2 = w' Sigma w = {var_P:.8f}")
print(f"  sigma_P               = {sigma_P:.6f}  = {sigma_P * 100:.2f} % p. a.")
print(f"  niedrigste Einzelvolatilitaet im Universum "
      f"= {sigma.min() * 100:.2f} % ({sigma.idxmin()})")
print(f"\n  ---> Das Portfolio schwankt weniger als JEDER seiner 15 Bestandteile.")

# Die Doppelsumme einmal aufgeschluesselt: woher kommt die Varianz?
diagonale = sum(w[i] * w[i] * Sigma.values[i, i] for i in range(N))
print(f"\n  Aufschluesselung der Doppelsumme w' Sigma w:")
print(f"    aus den {N} Varianzen (i = j)            {diagonale:.8f}   "
      f"({diagonale / var_P * 100:4.1f} %)")
print(f"    aus den {N * (N - 1)} Kovarianzen (i != j)      "
      f"{var_P - diagonale:.8f}   ({(var_P - diagonale) / var_P * 100:4.1f} %)")
print(f"  ---> Der weit groessere Teil des Portfoliorisikos steckt in den")
print(f"       Kreuztermen. Genau davon handelt § 2.2.")


# ────────────────────────────────────────────────────────────────────────
# Schritt 5 — Gegenprobe
# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 5 — Gegenprobe: Formel gegen direkte Berechnung")

# Statt ueber die Formel kann man das Portfolio auch direkt bilden: jeden Tag
# den Durchschnitt der 15 Renditen nehmen und daraus Mittelwert und
# Standardabweichung berechnen. Kommt dasselbe heraus, stimmt Formel (2.3).
portfolio_taeglich = renditen.mean(axis=1)
mu_direkt = portfolio_taeglich.mean() * HANDELSTAGE
sigma_direkt = portfolio_taeglich.std() * np.sqrt(HANDELSTAGE)

print(f"                      ueber Formel (2.3)     direkt aus der Reihe")
print(f"    mu_P              {mu_P * 100:12.6f} %   {mu_direkt * 100:14.6f} %")
print(f"    sigma_P           {sigma_P * 100:12.6f} %   {sigma_direkt * 100:14.6f} %")
print(f"\n    Abweichung        {abs(mu_P - mu_direkt):12.2e}     "
      f"{abs(sigma_P - sigma_direkt):14.2e}")
print("\n  ---> Beide Wege fuehren zum selben Ergebnis. Formel (2.3) ist keine")
print("       Naeherung, sondern eine Umformung: sie rechnet dasselbe anders.")

trennlinie("ZUSAMMENFASSUNG — die Zahlen fuer § 2.1")
print(f"  Zeitraum                                {start.date()} bis {ende.date()}")
print(f"  Handelstage                             {len(renditen)}")
print(f"  Renditen der Einzeltitel p. a.          "
      f"{mu.min() * 100:.1f} % bis {mu.max() * 100:.1f} %")
print(f"  Volatilitaeten der Einzeltitel p. a.    "
      f"{sigma.min() * 100:.1f} % bis {sigma.max() * 100:.1f} %")
print(f"  1/N-Portfolio: Rendite (arithmetisch)   {mu_P * 100:.2f} % p. a.")
print(f"  1/N-Portfolio: Volatilitaet             {sigma_P * 100:.2f} % p. a.")
print()
