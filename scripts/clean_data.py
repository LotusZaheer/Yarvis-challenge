"""
Script 02 - Limpieza y normalización del DataFrame de llamadas.
Entrada : pl.DataFrame validado (salida de load_data.py)
Salida  : pl.DataFrame limpio + data/processed/calls_clean.csv
"""

from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "calls_clean.csv"

# ---------------------------------------------------------------------------
# Campos a extraer de post_call_analysis (prefijo pca_)
# ---------------------------------------------------------------------------
PCA_FIELDS = [
    "resumen",
    "sentimiento",
    "razon_churn",
    "posible_recuperacion",
    "no_recarga",
    "dificultad_tecnica",
]


# ---------------------------------------------------------------------------
# Funciones privadas (cada una recibe y retorna pl.DataFrame)
# ---------------------------------------------------------------------------


def _parse_datetime(df: pl.DataFrame) -> pl.DataFrame:
    """Convierte executed_at a Datetime y extrae features temporales."""
    df = df.with_columns(
        pl.col("executed_at")
        .str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False)
        .alias("executed_at")
    )
    df = df.with_columns(
        pl.col("executed_at").dt.hour().alias("hour"),
        # weekday: lunes=1, domingo=7 (ISO), se deja como entero para EDA
        pl.col("executed_at").dt.weekday().alias("day_of_week"),
        pl.col("executed_at").dt.date().alias("date"),
    )
    return df


def _normalize_duration(df: pl.DataFrame) -> pl.DataFrame:
    """Castea duration_ms a Float64, agrega duration_sec y flag de outlier."""
    df = df.with_columns(
        pl.col("duration_ms").cast(pl.Float64, strict=False).alias("duration_ms")
    )
    df = df.with_columns(
        (pl.col("duration_ms") / 1000.0).alias("duration_sec")
    )
    # Percentil 99 calculado sobre valores no nulos
    p99 = df["duration_ms"].drop_nulls().quantile(0.99)
    df = df.with_columns(
        (pl.col("duration_ms") > p99).fill_null(False).alias("duration_outlier")
    )
    return df


def _normalize_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza columnas categóricas: tipos, trim y lowercase donde aplica."""
    df = df.with_columns(
        # campaign_id es identificador, no numérico
        pl.col("campaign_id").cast(pl.Utf8).alias("campaign_id"),
        # nombre de campaña: solo trim
        pl.col("name").str.strip_chars().alias("name"),
        # razón de desconexión: trim + lowercase + null → "unknown"
        pl.col("disconnected_reason")
        .str.strip_chars()
        .str.to_lowercase()
        .fill_null("unknown")
        .alias("disconnected_reason"),
    )
    return df


_PCA_DTYPE = pl.Struct({
    "resumen": pl.Utf8,
    "operador": pl.Utf8,
    "no_recarga": pl.Utf8,
    "razon_churn": pl.Utf8,
    "sentimiento": pl.Utf8,
    "recarga_servicio": pl.Utf8,
    "uso_del_servicio": pl.Utf8,
    "dificultad_tecnica": pl.Utf8,
    "posible_recuperacion": pl.Utf8,
})


def _parse_post_call_analysis(df: pl.DataFrame) -> pl.DataFrame:
    """Desempaqueta el JSON de post_call_analysis en columnas pca_*."""
    df = df.with_columns(
        pl.col("post_call_analysis").str.json_decode(_PCA_DTYPE).alias("_pca_struct")
    )
    # Extraer cada campo como columna independiente; strings vacíos → null
    df = df.with_columns(
        [
            pl.col("_pca_struct")
            .struct.field(field)
            .replace("", None)
            .alias(f"pca_{field}")
            for field in PCA_FIELDS
        ]
    )
    df = df.drop("_pca_struct")
    return df


def _parse_transcript(df: pl.DataFrame) -> pl.DataFrame:
    """Convierte el transcript JSON a texto plano y agrega longitud."""
    df = df.with_columns(
        pl.col("transcript")
        .str.json_decode(pl.List(pl.Utf8))  # List[Str]
        .list.join("\n")                     # texto plano continuo
        .str.strip_chars()
        .alias("transcript_text")
    )
    df = df.with_columns(
        pl.col("transcript_text").str.len_chars().alias("transcript_length")
    )
    return df


def _add_inconsistency_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Marca registros donde connected != call_completed."""
    df = df.with_columns(
        (pl.col("connected").fill_null(False) != pl.col("call_completed"))
        .alias("inconsistency_flag")
    )
    return df


def _validate(df_original: pl.DataFrame, df_clean: pl.DataFrame) -> None:
    """Valida el DataFrame limpio e imprime advertencias (no interrumpe el pipeline)."""
    # 1. Row count invariante
    if df_original.height != df_clean.height:
        print(f"[WARN] Filas originales: {df_original.height:,} | Filas limpias: {df_clean.height:,}")

    # 2. Columnas nuevas esperadas
    expected_new = [
        "hour", "day_of_week", "date",
        "duration_sec", "duration_outlier",
        "transcript_text", "transcript_length",
        "inconsistency_flag",
        *[f"pca_{f}" for f in PCA_FIELDS],
    ]
    missing_cols = [c for c in expected_new if c not in df_clean.columns]
    if missing_cols:
        print(f"[WARN] Columnas faltantes tras limpieza: {missing_cols}")

    # 3. Nulos en executed_at (no deberían existir)
    nulls_dt = df_clean["executed_at"].null_count()
    if nulls_dt > 0:
        print(f"[WARN] {nulls_dt:,} nulos en executed_at tras parseo datetime")

    # 4. Conteo de flags de calidad
    incons = df_clean["inconsistency_flag"].sum()
    outliers = df_clean["duration_outlier"].sum()
    print(f"[INFO] Registros inconsistentes (connected != call_completed): {incons:,}")
    print(f"[INFO] Outliers de duración (>p99): {outliers:,}")

    # 5. Valores válidos en pca_sentimiento
    if "pca_sentimiento" in df_clean.columns:
        valid_sentimientos = {"neutral", "negativo", "positivo", None}
        found = set(df_clean["pca_sentimiento"].unique().to_list())
        unexpected = found - valid_sentimientos
        if unexpected:
            print(f"[WARN] Valores inesperados en pca_sentimiento: {unexpected}")
        else:
            sentimiento_dist = df_clean["pca_sentimiento"].value_counts(sort=True)
            print(f"[INFO] Distribución pca_sentimiento:\n{sentimiento_dist}")


def _export(df: pl.DataFrame) -> None:
    """Exporta el DataFrame limpio a data/processed/calls_clean.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(CLEAN_CSV)
    print(f"[INFO] Exportado: {CLEAN_CSV.name} ({df.height:,} filas, {len(df.columns)} columnas)")


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------


def clean(df: pl.DataFrame) -> pl.DataFrame:
    """
    Limpia y normaliza el DataFrame crudo.
    No muta la entrada; retorna un nuevo DataFrame enriquecido.
    """
    df_original = df
    df = _parse_datetime(df)
    df = _normalize_duration(df)
    df = _normalize_categoricals(df)
    df = _parse_post_call_analysis(df)
    df = _parse_transcript(df)
    df = _add_inconsistency_flag(df)
    _validate(df_original, df)
    _export(df)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from load_data import load_raw

    df_raw = load_raw()
    df_clean = clean(df_raw)
    print("\n[OK] Limpieza completada.\n")
    print(df_clean.head(3))
