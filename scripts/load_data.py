"""
Script 01 - Carga y validación inicial del CSV de llamadas.
Entrada : data/raw/data_calls.csv
Salida  : pl.DataFrame validado (listo para etapa de limpieza)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from utils.paths import RAW_CSV

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
SEPARATOR = ","
NULL_VALUES = ["NULL", "NaN", "nan", ""]

# Esquema esperado: nombre original → nombre normalizado (snake_case)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_columns(df: pl.DataFrame) -> None:
    """Verifica que el CSV contenga todas las columnas esperadas."""
    missing = set(COLUMN_RENAME.keys()) - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Columnas faltantes en el CSV: {missing}")


def _add_derived_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega columnas calculadas a partir de las existentes."""
    return df.with_columns(
        pl.col("call_url").is_not_null().alias("call_completed")
    )


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def load_raw(path: Path = RAW_CSV) -> pl.DataFrame:
    """Lee el CSV, renombra columnas a snake_case y valida estructura."""
    if not path.exists():
        sys.exit(f"[ERROR] Archivo no encontrado: {path}")

    df = pl.read_csv(path, separator=SEPARATOR, null_values=NULL_VALUES)

    _validate_columns(df)
    df = df.rename(COLUMN_RENAME)
    df = _add_derived_columns(df)

    print(f"[INFO] Registros cargados: {df.height:,}")
    return df


if __name__ == "__main__":
    df = load_raw()
    print("\n[OK] Carga completada.\n")
    print(df.head(3))
