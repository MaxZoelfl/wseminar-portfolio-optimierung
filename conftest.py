"""pytest-Setup: headless Matplotlib + Projektwurzel auf sys.path."""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(__file__))
