"""Helper de cache CSV para el pipeline — fuente única de lógica de cache.

Uso:
    from utils.cache import load_csv_cache

    cached = load_csv_cache(CACHE_PATH, expected_rows=df.height, expected_cols=["col_a"])
    if cached is not None:
        return cached   # cache válido, evitar recálculo
    # ... recalcular ...
"""

from pathlib import Path

import polars as pl


def load_csv_cache(
    path: Path,
    expected_rows: int | None = None,
    expected_cols: list[str] | None = None,
    min_size_kb: int = 0,
    min_rows: int = 0,
    infer_schema_length: int = 1000,
) -> pl.DataFrame | None:
    """Carga un CSV de cache si existe y cumple todas las condiciones.

    Args:
        path: ruta al archivo CSV de cache.
        expected_rows: invalida el cache si las filas no coinciden exactamente.
        expected_cols: lista de columnas que deben estar presentes.
        min_size_kb: tamaño mínimo en KB que debe tener el archivo en disco.
        min_rows: número mínimo de filas requeridas (útil para caches que
            deben representar datasets grandes, como calls_clean.csv).
        infer_schema_length: filas usadas por Polars para inferir el schema.

    Returns:
        pl.DataFrame del cache si es válido, None si hay que recalcular.
    """
    if not path.exists():
        return None

    if min_size_kb and path.stat().st_size < min_size_kb * 1024:
        print(f"[WARN] Cache inválido (tamaño < {min_size_kb} KB), recalculando...")
        return None

    df = pl.read_csv(path, infer_schema_length=infer_schema_length)

    if expected_rows is not None and df.height != expected_rows:
        print(
            f"[WARN] Cache inválido (filas: {df.height:,} vs esperadas: {expected_rows:,}),"
            " recalculando..."
        )
        return None

    if min_rows and df.height < min_rows:
        print(f"[WARN] Cache inválido (filas: {df.height:,} < mínimo {min_rows:,}), recalculando...")
        return None

    if expected_cols and not all(c in df.columns for c in expected_cols):
        missing = [c for c in expected_cols if c not in df.columns]
        print(f"[WARN] Cache inválido (columnas faltantes: {missing}), recalculando...")
        return None

    print(f"[INFO] Cache encontrado: {path.name} ({df.height:,} filas)")
    return df

