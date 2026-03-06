"""
Script 02 - Limpieza y normalización del DataFrame de llamadas.
Entrada : pl.DataFrame validado (salida de load_data.py)
Salida  : pl.DataFrame limpio + data/processed/calls_clean.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from utils.cache import load_csv_cache
from utils.paths import PROCESSED_DIR, CLEAN_CSV
from utils.text import strip_accents

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATETIME_FMT = "%Y-%m-%d %H:%M:%S%.f"
MS_TO_SEC = 1000.0
P99_QUANTILE = 0.99
IQR_MULTIPLIER = 1.5
CACHE_MIN_ROWS = 25000

PCA_FIELDS = [
    "resumen",
    "sentimiento",
    "razon_churn",
    "posible_recuperacion",
    "no_recarga",
    "dificultad_tecnica",
]

EXPECTED_NEW_COLS = [
    "hour", "day_of_week", "day_of_month", "date",
    "duration_sec", "duration_outlier",
    "transcript_text", "transcript_length",
    "inconsistency_flag",
    *[f"pca_{f}" for f in PCA_FIELDS],
]


# ---------------------------------------------------------------------------
# Funciones privadas (cada una recibe y retorna pl.DataFrame)
# ---------------------------------------------------------------------------


DIAS_SEMANA = {1: "lu", 2: "ma", 3: "mi", 4: "ju", 5: "vi", 6: "sa", 7: "do"}


def _parse_datetime(df: pl.DataFrame) -> pl.DataFrame:
    """Convierte executed_at a Datetime y extrae features temporales."""
    return df.with_columns(
        pl.col("executed_at")
        .str.to_datetime(DATETIME_FMT, strict=False)
        .alias("executed_at")
    ).with_columns(
        pl.col("executed_at").dt.hour().alias("hour"),
        pl.col("executed_at").dt.day().alias("day_of_month"),
        pl.col("executed_at").dt.strftime("%d/%m/%Y").alias("date"),
        pl.col("executed_at")
        .dt.weekday()
        .replace_strict(DIAS_SEMANA)
        .alias("day_of_week"),
    )

def _normalize_duration(df: pl.DataFrame) -> pl.DataFrame:
    """Castea duration_ms a Float64, agrega duration_sec y flag de outlier."""
    df = df.with_columns(
        pl.col("duration_ms").cast(pl.Float64, strict=False).alias("duration_ms")
    ).with_columns(
        (pl.col("duration_ms") / MS_TO_SEC).alias("duration_sec")
    )
    p99 = df["duration_ms"].drop_nulls().quantile(P99_QUANTILE)
    return df.with_columns(
        (pl.col("duration_ms") > p99).fill_null(False).alias("duration_outlier")
    )


def _normalize_pca_value(value: str, space_char: str = "_") -> str:
    """Normaliza un valor PCA: strip, lowercase, sin tildes, espacios→space_char."""
    if not value:
        return None
    normalized = strip_accents(value.strip().lower())
    normalized = normalized.replace(" ", space_char)
    return normalized if normalized else None


def _normalize_pca_razon_churn(value: str) -> str:
    """Normaliza pca_razon_churn con alias específicos."""
    normalized = _normalize_pca_value(value, space_char="_")
    if not normalized:
        return None

    # Alias y unificación conceptual
    alias_map = {
        "no_supo_recargar": "no_sabe_recargar",  # unificar variantes
        "no_recarga": "no_sabe_recargar",
        "fallos_(caida_del_servicio_o_servicio_mal_instalado)": "fallas_(caida_del_servicio_o_servicio_mal_instalado)",
    }
    return alias_map.get(normalized, normalized)





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


# _PCA_DTYPE incluye TODOS los campos del JSON para una deserialización fiel.
# Solo los campos en PCA_FIELDS se extraen como columnas en el DataFrame.
# Los campos extra (operador, recarga_servicio, uso_del_servicio) se descartan.
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
    return df.with_columns(
        pl.col("transcript")
        .str.json_decode(pl.List(pl.Utf8))
        .list.join("\n")
        .str.strip_chars()
        .alias("transcript_text")
    ).with_columns(
        pl.col("transcript_text").str.len_chars().alias("transcript_length")
    )


def _normalize_pca_fields(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza columnas pca_* categóricas: tildes, espacios→guiones, lowercase."""
    cols_to_normalize = {
        "pca_razon_churn": _normalize_pca_razon_churn,
        "pca_posible_recuperacion": _normalize_pca_value,   # sin alias: misma lógica base
    }
    for col, normalize_fn in cols_to_normalize.items():
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .map_elements(normalize_fn, return_dtype=pl.Utf8)
                .alias(col)
            )
    return df


def _add_inconsistency_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Marca registros donde connected != call_completed."""
    return df.with_columns(
        (pl.col("connected").fill_null(False) != pl.col("call_completed"))
        .alias("inconsistency_flag")
    )


def _validate(df_original: pl.DataFrame, df_clean: pl.DataFrame) -> None:
    """Valida el DataFrame limpio e imprime advertencias (no interrumpe el pipeline)."""
    if df_original.height != df_clean.height:
        print(f"[WARN] Filas originales: {df_original.height:,} | Filas limpias: {df_clean.height:,}")

    missing_cols = [c for c in EXPECTED_NEW_COLS if c not in df_clean.columns]
    if missing_cols:
        print(f"[WARN] Columnas faltantes tras limpieza: {missing_cols}")

    nulls_dt = df_clean["executed_at"].null_count()
    if nulls_dt > 0:
        print(f"[WARN] {nulls_dt:,} nulos en executed_at tras parseo datetime")

    print(f"[INFO] Registros inconsistentes (connected != call_completed): {df_clean['inconsistency_flag'].sum():,}")
    print(f"[INFO] Outliers de duracion (>p99): {df_clean['duration_outlier'].sum():,}")

    _validate_sentimiento(df_clean)


def _validate_sentimiento(df: pl.DataFrame) -> None:
    """Valida valores de pca_sentimiento."""
    if "pca_sentimiento" not in df.columns:
        return
    valid = {"neutral", "negativo", "positivo", None}
    found = set(df["pca_sentimiento"].unique().to_list())
    unexpected = found - valid
    if unexpected:
        print(f"[WARN] Valores inesperados en pca_sentimiento: {unexpected}")
    else:
        print(f"[INFO] Distribucion pca_sentimiento:\n{df['pca_sentimiento'].value_counts(sort=True)}")


def _report_filter(n_before: int, n_after: int, steps: list[tuple[str, int]]) -> None:
    """Imprime resumen de registros filtrados."""
    n_filtered = n_before - n_after
    if n_filtered > 0:
        pct = n_filtered / n_before * 100
        print(f"[INFO] Filtro aplicado: {n_filtered:,} registros removidos ({pct:.2f}%)")
        for label, count in steps:
            print(f"       → {label}: {count}")


def _filter_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Filtra hora 0, campaña 'prueba' y outliers de duración (IQR)."""
    n_before = df.height
    steps = []

    df = df.filter(pl.col("hour") != 0)
    steps.append(("hora 0", n_before - df.height))

    prev = df.height
    if "campaign_type" in df.columns:
        df = df.filter(pl.col("campaign_type") != "prueba")
    steps.append(("campaña 'prueba'", prev - df.height))

    q1 = df.select(pl.col("duration_sec").quantile(0.25)).item()
    q3 = df.select(pl.col("duration_sec").quantile(0.75)).item()
    iqr = q3 - q1
    prev = df.height
    df = df.filter(
        pl.col("duration_sec").is_between(q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr)
    )
    steps.append(("outliers duración", prev - df.height))

    _report_filter(n_before, df.height, steps)
    return df


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
    Si calls_clean.csv ya existe y tiene las columnas esperadas, lo carga directamente.
    No muta la entrada; retorna un nuevo DataFrame enriquecido.
    """
    _cache_check_cols = ["hour", "day_of_week", "duration_sec", "transcript_text", "inconsistency_flag"]

    cached = load_csv_cache(
        CLEAN_CSV,
        expected_cols=_cache_check_cols,
        min_rows=CACHE_MIN_ROWS,
        infer_schema_length=5000,
    )
    if cached is not None:
        return cached

    df_original = df
    df = _parse_datetime(df)
    df = _normalize_duration(df)
    df = _normalize_categoricals(df)
    df = _parse_post_call_analysis(df)
    df = _normalize_pca_fields(df)
    df = _parse_transcript(df)
    df = _add_inconsistency_flag(df)
    df = _filter_outliers(df)
    _validate(df_original, df)
    _export(df)
    return df


if __name__ == "__main__":
    from scripts.load_data import load_raw

    df_raw = load_raw()
    df_clean = clean(df_raw)
    print("\n[OK] Limpieza completada.\n")
    print(df_clean.head(3))
