"""Rutas centralizadas del proyecto — fuente única de verdad."""

from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures"

RAW_CSV   = RAW_DIR / "data_calls.csv"
CLEAN_CSV = PROCESSED_DIR / "calls_clean.csv"
