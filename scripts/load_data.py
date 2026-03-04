"""
Script 01 - Carga y validación inicial del CSV de llamadas.
Entrada : data/raw/data_calls.csv
Salida  : pd.DataFrame validado (listo para etapa de limpieza)
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "data_calls.csv"

# ---------------------------------------------------------------------------
# Esquema esperado: nombre original → nombre normalizado (snake_case)
# ---------------------------------------------------------------------------
COLUMN_RENAME = {
    "Campaign Id": "campaign_id",
    "Name": "name",
    "Target Id": "target_id",
    "Call URL": "call_url",
    "Connected": "connected",
    "Disconnected Reason": "disconnected_reason",
    "Duration (ms)": "duration_ms",
    "Transcript": "transcript",
    "Post Call Analysis": "post_call_analysis",
    "Executed At": "executed_at",
}


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Lee el CSV con separador coma, renombra columnas a snake_case
    y ejecuta validaciones básicas de estructura.
    """
    if not path.exists():
        sys.exit(f"[ERROR] Archivo no encontrado: {path}")

    df = pd.read_csv(path, sep=",", encoding="utf-8")

    # --- Validar columnas esperadas ---
    missing = set(COLUMN_RENAME.keys()) - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Columnas faltantes en el CSV: {missing}")

    df = df.rename(columns=COLUMN_RENAME)

    # --- Reporte rápido ---
    print(f"[INFO] Registros cargados : {len(df):,}")
    print(f"[INFO] Columnas           : {list(df.columns)}")
    print("\n[INFO] Tipos de dato:")
    print(df.dtypes.to_string())
    print("\n[INFO] Valores nulos por columna:")
    print(df.isnull().sum().to_string())
    print(f"\n[INFO] Conectadas (True)  : {df['connected'].sum() if df['connected'].dtype == bool else (df['connected'] == True).sum():,}")

    return df


if __name__ == "__main__":
    df = load_raw()
    print("\n[OK] Carga completada.\n")
    print(df.head(3).to_string())
