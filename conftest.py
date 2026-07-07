"""pytest-Setup: headless Matplotlib + Projektwurzel auf sys.path.

FÜR EINSTEIGER: pytest ist das Test-Werkzeug des Projekts — es führt die
Prüf-Programme im Ordner tests/ aus. Eine Datei namens conftest.py liest
pytest automatisch VOR allen Tests ein. Hier werden zwei Dinge vorbereitet:
  1. MPLBACKEND=Agg — Diagramme nur unsichtbar im Speicher zeichnen, damit
     während der Tests keine Fenster aufpoppen (und Tests auch auf Servern
     ohne Bildschirm laufen).
  2. Der Projektordner wird an den Anfang des Python-Suchpfads gestellt,
     damit die Tests ``import portfolio`` finden, egal von wo man sie startet.
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(__file__))
