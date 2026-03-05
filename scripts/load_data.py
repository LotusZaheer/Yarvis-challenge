"""
Script 01 - Carga y validación inicial del CSV de llamadas.
Entrada : data/raw/data_calls.csv
Salida  : pd.DataFrame validado (listo para etapa de limpieza)
"""

import sys
from pathlib import Path

import polars as pl

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


def load_raw(path: Path = RAW_CSV) -> pl.DataFrame:
    """
    Lee el CSV con separador coma, renombra columnas a snake_case
    y ejecuta validaciones básicas de estructura.
    """
    if not path.exists():
        sys.exit(f"[ERROR] Archivo no encontrado: {path}")

    # Carga con polars indicando que "NULL" representa valor nulo
    df = pl.read_csv(path, separator=",", null_values=["NULL", "NaN", "nan", ""])

    # --- Validar columnas esperadas ---
    missing = set(COLUMN_RENAME.keys()) - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Columnas faltantes en el CSV: {missing}")

    # Renombrar
    df = df.rename(COLUMN_RENAME)

    # --- Columna derivada: llamada efectuada ---
    # Si call_url tiene valor, la llamada logro conectar; si es nulo, no conecto
    df = df.with_columns(
        pl.col("call_url").is_not_null().alias("call_completed")
    )


    # --- Reporte rápido ---
    print(f"[INFO] Registros cargados : {df.height:,}")
    print(f"[INFO] Columnas           : {df.columns}")
    
    print("\n[INFO] Tipos de dato:")
    for col_name, dtype in zip(df.columns, df.dtypes):
        print(f"{col_name}: {dtype}")
        
    print("\n[INFO] Valores nulos por columna:")
    nulls = df.null_count()
    for col in df.columns:
        print(f"{col}: {nulls[col][0]}")
        
    conectadas = df.filter(pl.col("connected") == True).height
    print(f"\n[INFO] Conectadas (True)  : {conectadas:,}")

    return df


if __name__ == "__main__":
    df = load_raw()
    print("\n[OK] Carga completada.\n")
    print(df.head(3))
