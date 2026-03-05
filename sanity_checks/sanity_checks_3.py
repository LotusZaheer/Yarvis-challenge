"""
Sanity Check 3 — Análisis de registros eliminados.
Compara CSV original vs CSV limpio para identificar y categorizar
qué registros fueron eliminados y sus características.

Escribe resultados en: data/interim/sc3_eliminated_analysis.txt
Genera gráficos en: data/interim/sc3_*.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "data_calls.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "calls_clean.csv"
OUT_DIR = PROJECT_ROOT / "data" / "interim"
OUT_FILE = OUT_DIR / "sc3_eliminated_analysis.txt"

# Mapeo de columnas para normalización
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe(series: pl.Series) -> dict:
    """Estadísticas de una serie numérica."""
    filtered = series.drop_nulls()
    if filtered.is_empty():
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0, "min": 0, "max": 0}
    try:
        arr = filtered.cast(pl.Float64).to_numpy()
        arr = np.asarray(arr, dtype=float)
    except (ValueError, TypeError):
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0, "min": 0, "max": 0}
    if len(arr) == 0:
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0, "min": 0, "max": 0}
    return {
        "n":      len(arr),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "q25":    float(np.percentile(arr, 25)),
        "q75":    float(np.percentile(arr, 75)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


def _section(lines: list, title: str):
    """Formato de sección en el reporte."""
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)


def _normalize_raw_df(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza columnas del CSV raw y crea derivadas necesarias."""
    import re

    # Renombrar si tiene columnas originales
    if "Campaign Id" in df.columns:
        df = df.rename(COLUMN_RENAME)

    # Crear columnas necesarias si faltan
    if "duration_sec" not in df.columns:
        if "duration_ms" in df.columns:
            df = df.with_columns(
                (pl.col("duration_ms").cast(pl.Float64, strict=False) / 1000).alias("duration_sec")
            )

    if "hour" not in df.columns and "executed_at" in df.columns:
        try:
            df = df.with_columns(
                pl.col("executed_at")
                .str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False)
                .dt.hour()
                .alias("hour")
            )
        except Exception:
            df = df.with_columns(pl.lit(0).alias("hour"))

    if "campaign_type" not in df.columns and "name" in df.columns:
        def classify(name):
            if not name:
                return "otro"
            n = name.lower()
            patterns = [
                (r"churn|posibles churn|early churn|retenci", "churn_prevention"),
                (r"upsell|tv digital", "upsell"),
                (r"sales", "sales"),
                (r"suspendido|2do pago|1er pago|preventiva|cobranza", "cobranza"),
                (r"aviso", "aviso"),
                (r"servicio al cliente", "servicio_cliente"),
                (r"promo|rel.mpago", "promo"),
                (r"prueba", "prueba"),
            ]
            for pat, lbl in patterns:
                if re.search(pat, n):
                    return lbl
            return "otro"
        df = df.with_columns(
            pl.col("name").map_elements(classify, return_dtype=pl.Utf8).alias("campaign_type")
        )

    if "sentiment_own" not in df.columns:
        df = df.with_columns(pl.lit("neutral").alias("sentiment_own"))

    return df


def _classify_campaign(name):
    """Clasifica campaña según patrones."""
    import re
    _CAMPAIGN_PATTERNS = [
        (r"churn|posibles churn|early churn|retenci", "churn_prevention"),
        (r"upsell|tv digital", "upsell"),
        (r"sales", "sales"),
        (r"suspendido|2do pago|1er pago|preventiva|cobranza", "cobranza"),
        (r"aviso", "aviso"),
        (r"servicio al cliente", "servicio_cliente"),
        (r"promo|rel.mpago", "promo"),
        (r"prueba", "prueba"),
    ]
    if not name:
        return "otro"
    n = name.lower()
    for pat, lbl in _CAMPAIGN_PATTERNS:
        if re.search(pat, n):
            return lbl
    return "otro"


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def run(df_raw: pl.DataFrame, df_clean: pl.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    # Normalizar ambos dataframes
    df_raw = _normalize_raw_df(df_raw)
    df_clean = _normalize_raw_df(df_clean)

    # Calcular umbrales de outliers usando método IQR (igual a clean_data.py)
    # Primero, filtrar hora 0 y campaña prueba para calcular IQR sobre datos limpios
    df_for_iqr = df_raw.filter(
        (pl.col("hour") != 0) &
        (pl.col("campaign_type") != "prueba")
    )

    try:
        q1 = float(df_for_iqr.select(pl.col("duration_sec").quantile(0.25)).item())
        q3 = float(df_for_iqr.select(pl.col("duration_sec").quantile(0.75)).item())
        iqr = q3 - q1
        outlier_lower = q1 - 1.5 * iqr
        outlier_upper = q3 + 1.5 * iqr
    except Exception:
        outlier_lower = -float('inf')
        outlier_upper = float('inf')

    # Identificar cuáles fueron eliminados (comparar IDs o índices)
    # Usamos las columnas de "id" o creamos un hash
    if "call_id" in df_raw.columns and "call_id" in df_clean.columns:
        ids_raw = set(df_raw["call_id"].to_list())
        ids_clean = set(df_clean["call_id"].to_list())
        ids_eliminated = ids_raw - ids_clean
        df_eliminated = df_raw.filter(pl.col("call_id").is_in(list(ids_eliminated)))
    else:
        # Alternativa: usar índice o hacer un left join
        df_with_idx = df_raw.with_row_index("idx")
        df_clean_with_idx = df_clean.with_row_index("idx")
        idx_clean = set(df_clean_with_idx["idx"].to_list())
        df_eliminated = df_with_idx.filter(~pl.col("idx").is_in(list(idx_clean))).drop("idx")

    lines.append("SANITY CHECK 3 — Análisis de registros eliminados")
    lines.append(f"Dataset original  : {df_raw.height:,} registros")
    lines.append(f"Dataset limpio    : {df_clean.height:,} registros")
    lines.append(f"Registros eliminados: {df_eliminated.height:,} ({df_eliminated.height / df_raw.height:.1%})")

    # -----------------------------------------------------------------------
    # 1. Categorizar eliminados
    # -----------------------------------------------------------------------
    _section(lines, "1. CATEGORIZACIÓN DE ELIMINADOS")
    lines.append("Desglose por causa de eliminación según filtros de limpieza\n")

    # Crear categorías de eliminación
    def categorize_elimination(row):
        """Asigna categoría de eliminación según reglas de limpieza."""
        if row.get("hour") == 0:
            return "hora_0"
        if row.get("campaign_type") == "prueba":
            return "campaign_prueba"
        dur = row.get("duration_sec")
        if dur is not None and (dur < outlier_lower or dur > outlier_upper):
            return "outlier_duration"
        return "desconectadas_o_incompletas"

    df_elim = df_eliminated.to_dicts()
    categories = [categorize_elimination(row) for row in df_elim]

    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Orden personalizado de presentación
    order = ["desconectadas_o_incompletas", "outlier_duration", "hora_0", "campaign_prueba"]
    for cat in order:
        if cat in cat_counts:
            count = cat_counts[cat]
            pct = count / len(categories) * 100
            lines.append(f"  {cat:30s}: {count:7,}  ({pct:5.1f}%)")

    lines.append(f"\n  Nota: 'desconectadas_o_incompletas' = registros no conectados o sin datos suficientes")

    # -----------------------------------------------------------------------
    # 2. Distribución temporal de eliminados
    # -----------------------------------------------------------------------
    _section(lines, "2. DISTRIBUCIÓN TEMPORAL (por HORA)")
    lines.append("Análisis de cuáles horas contribuyeron a la eliminación\n")

    hour_dist = (
        df_raw.group_by("hour")
        .agg([
            pl.len().alias("total"),
            (pl.col("hour") == 0).sum().alias("eliminados"),
        ])
        .with_columns(
            (pl.col("total") - pl.col("eliminados")).alias("mantenidos")
        )
        .with_columns(
            (pl.col("eliminados") / pl.col("total") * 100).round(1).alias("pct_eliminados")
        )
        .sort("hour")
    )

    lines.append(f"{'Hora':>5}  {'Total':>10}  {'Eliminados':>12}  {'Mantenidos':>12}  {'% Elim.':>10}")
    lines.append("-" * 65)
    for row in hour_dist.iter_rows(named=True):
        h = row["hour"]
        total = row["total"]
        elim = row["eliminados"]
        mant = row["mantenidos"]
        pct = row["pct_eliminados"]
        marker = " ◄ TODOS" if pct == 100.0 else ""
        lines.append(f"{h:>5}  {total:>10,}  {elim:>12,}  {mant:>12,}  {pct:>9.1f}%{marker}")

    lines.append(f"\n  Nota: Únicamente la hora 0 (00:00–00:59) fue completamente eliminada")

    # -----------------------------------------------------------------------
    # 3. Distribución por campaña de eliminados
    # -----------------------------------------------------------------------
    _section(lines, "3. DISTRIBUCIÓN POR CAMPAÑA")
    lines.append("Análisis de qué campaña contribuyó más a la eliminación\n")

    campaign_dist = (
        df_raw.group_by("campaign_type")
        .agg([
            pl.len().alias("total"),
            (pl.col("hour") == 0).sum().alias("eliminados"),
        ])
        .with_columns(
            (pl.col("total") - pl.col("eliminados")).alias("mantenidos")
        )
        .with_columns(
            (pl.col("eliminados") / pl.col("total") * 100).round(1).alias("pct_eliminados")
        )
        .sort("eliminados", descending=True)
    )

    lines.append(f"{'Campaña':>25}  {'Total':>10}  {'Eliminados':>12}  {'Mantenidos':>12}  {'% Elim.':>10}")
    lines.append("-" * 75)
    for row in campaign_dist.iter_rows(named=True):
        camp = row["campaign_type"]
        total = row["total"]
        elim = row["eliminados"]
        mant = row["mantenidos"]
        pct = row["pct_eliminados"]
        marker = " ◄" if pct == 100.0 else ""
        lines.append(
            f"{str(camp):>25}  {total:>10,}  {elim:>12,}  {mant:>12,}  {pct:>9.1f}%{marker}"
        )

    lines.append(f"\n  Nota: Los registros eliminados son aquellos con hour==0")
    lines.append(f"  '% Elim.' = porcentaje de esa campaña que fue eliminado")

    # -----------------------------------------------------------------------
    # 4. Características de duración (eliminados vs mantenidos)
    # -----------------------------------------------------------------------
    _section(lines, "4. DURACIÓN DE LLAMADAS (conectadas)")

    df_elim_conn = df_eliminated.filter(pl.col("connected") == True)
    df_clean_conn = df_clean.filter(pl.col("connected") == True)

    for label, sub in [("Eliminados", df_elim_conn), ("Mantenidos", df_clean_conn)]:
        d = _describe(sub["duration_sec"])
        lines.append(
            f"  {label:12s}: n={d['n']:,}  "
            f"media={d['mean']:.1f}s  mediana={d['median']:.1f}s  "
            f"IQR=[{d['q25']:.0f}, {d['q75']:.0f}]  "
            f"rango=[{d['min']:.0f}, {d['max']:.0f}]"
        )

    # Mann-Whitney entre eliminados y mantenidos
    elim_dur = df_elim_conn["duration_sec"].drop_nulls().cast(pl.Float64).to_list()
    clean_dur = df_clean_conn["duration_sec"].drop_nulls().cast(pl.Float64).to_list()

    if len(elim_dur) >= 2 and len(clean_dur) >= 2:
        try:
            u_stat, p_val = stats.mannwhitneyu(elim_dur, clean_dur, alternative="two-sided")
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            lines.append(f"\n  Mann-Whitney (eliminados vs mantenidos): U={u_stat:.0f}, p={p_val:.4e} {sig}")
        except Exception as e:
            lines.append(f"\n  Mann-Whitney: error — {e}")

    # -----------------------------------------------------------------------
    # 5. Razón de desconexión de eliminados conectados
    # -----------------------------------------------------------------------
    _section(lines, "5. RAZÓN DE DESCONEXIÓN (eliminados conectados)")

    if df_elim_conn.height > 0:
        dist = (
            df_elim_conn.group_by("disconnected_reason")
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / df_elim_conn.height * 100).round(1).alias("pct"))
            .sort("n", descending=True)
        )
        lines.append(f"  {df_elim_conn.height:,} llamadas conectadas eliminadas:")
        lines.append(f"{'Razón':>30}  {'Cantidad':>10}  {'%':>8}")
        lines.append("-" * 50)
        for row in dist.iter_rows(named=True):
            lines.append(f"  {str(row['disconnected_reason']):30s}  {row['n']:>10,}  {row['pct']:>7.1f}%")

    # -----------------------------------------------------------------------
    # 6. Sentimiento en eliminados conectados
    # -----------------------------------------------------------------------
    _section(lines, "6. SENTIMIENTO (eliminados conectados)")

    if df_elim_conn.height > 0 and "sentiment_own" in df_elim_conn.columns:
        dist = (
            df_elim_conn.group_by("sentiment_own")
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / df_elim_conn.height * 100).round(1).alias("pct"))
            .sort("n", descending=True)
        )
        lines.append(f"  {df_elim_conn.height:,} llamadas conectadas eliminadas:")
        lines.append(f"{'Sentimiento':>20}  {'Cantidad':>10}  {'%':>8}")
        lines.append("-" * 42)
        for row in dist.iter_rows(named=True):
            lines.append(f"  {str(row['sentiment_own']):20s}  {row['n']:>10,}  {row['pct']:>7.1f}%")

    # -----------------------------------------------------------------------
    # 7. Tasa de conexión
    # -----------------------------------------------------------------------
    _section(lines, "7. TASA DE CONEXIÓN")

    n_elim_total = df_eliminated.height
    n_elim_conn = df_eliminated.filter(pl.col("connected") == True).height
    rate_elim = n_elim_conn / n_elim_total if n_elim_total > 0 else 0

    n_clean_total = df_clean.height
    n_clean_conn = df_clean.filter(pl.col("connected") == True).height
    rate_clean = n_clean_conn / n_clean_total if n_clean_total > 0 else 0

    lines.append(f"  Eliminados: {rate_elim:.1%} conectadas ({n_elim_conn}/{n_elim_total})")
    lines.append(f"  Mantenidos: {rate_clean:.1%} conectadas ({n_clean_conn}/{n_clean_total})")

    # -----------------------------------------------------------------------
    # 8. Resumen por causa de eliminación
    # -----------------------------------------------------------------------
    _section(lines, "8. RESUMEN DETALLADO POR CAUSA")

    causa_hora0 = df_eliminated.filter(pl.col("hour") == 0)
    causa_prueba = df_eliminated.filter(pl.col("campaign_type") == "prueba")
    causa_outlier = df_eliminated.filter(
        (pl.col("duration_sec") < outlier_lower) |
        (pl.col("duration_sec") > outlier_upper)
    )

    for label, sub in [("Hora 0", causa_hora0), ("Campaign prueba", causa_prueba), ("Outliers duración", causa_outlier)]:
        if sub.height > 0:
            lines.append(f"\n  {label} ({sub.height:,} registros):")
            lines.append(f"    Conectadas: {sub.filter(pl.col('connected') == True).height} "
                        f"({sub.filter(pl.col('connected') == True).height / sub.height:.1%})")

            conn_sub = sub.filter(pl.col("connected") == True)
            if conn_sub.height > 0:
                d = _describe(conn_sub["duration_sec"])
                lines.append(f"    Duración conectadas: media={d['mean']:.1f}s, mediana={d['median']:.1f}s")
        else:
            lines.append(f"\n  {label}: (ninguno)")

    # -----------------------------------------------------------------------
    # Guardar reporte
    # -----------------------------------------------------------------------
    output = "\n".join(lines)
    OUT_FILE.write_text(output, encoding="utf-8")
    print(output)
    print(f"\n[INFO] Reporte guardado en: {OUT_FILE}")

    # -----------------------------------------------------------------------
    # Gráficos
    # -----------------------------------------------------------------------
    _create_plots(df_raw, df_clean, df_eliminated)


def _create_plots(df_raw, df_clean, df_eliminated):
    """Genera gráficos de análisis de eliminados."""

    # 1. Histograma de duración: eliminados vs mantenidos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    elim_conn = df_eliminated.filter(
        (pl.col("connected") == True) &
        (pl.col("duration_sec").is_not_null())
    )["duration_sec"].to_numpy()
    clean_conn = df_clean.filter(
        (pl.col("connected") == True) &
        (pl.col("duration_sec").is_not_null())
    )["duration_sec"].to_numpy()

    axes[0].hist(clean_conn, bins=50, alpha=0.7, color="green", edgecolor="black")
    axes[0].set_xlabel("Duración (segundos)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title(f"Mantenidos (n={len(clean_conn)})")
    axes[0].grid(alpha=0.3)

    axes[1].hist(elim_conn, bins=50, alpha=0.7, color="red", edgecolor="black")
    axes[1].set_xlabel("Duración (segundos)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"Eliminados (n={len(elim_conn)})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "sc3_duration_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico: sc3_duration_comparison.png")

    # 2. Distribución de eliminados por hora
    hour_dist = df_eliminated.group_by("hour").agg(pl.len().alias("n")).sort("hour")
    hours = [row["hour"] for row in hour_dist.iter_rows(named=True)]
    counts = [row["n"] for row in hour_dist.iter_rows(named=True)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hours, counts, color="coral", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Registros eliminados")
    ax.set_title(f"Distribución de eliminados por hora (total={len(hours)} horas con datos)")
    ax.set_xticks(range(0, 24))
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT_DIR / "sc3_eliminated_by_hour.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico: sc3_eliminated_by_hour.png")

    # 3. Distribución por campaña
    camp_dist = (
        df_eliminated.group_by("campaign_type")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    campaigns = [row["campaign_type"] for row in camp_dist.iter_rows(named=True)]
    counts = [row["n"] for row in camp_dist.iter_rows(named=True)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(campaigns, counts, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Registros eliminados")
    ax.set_title("Distribución de eliminados por campaña")
    ax.grid(alpha=0.3, axis="x")

    # Añadir etiquetas de cantidad
    for i, (camp, count) in enumerate(zip(campaigns, counts)):
        ax.text(count, i, f" {count:,}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sc3_eliminated_by_campaign.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico: sc3_eliminated_by_campaign.png")

    # 4. Gráfico de barras de categorías de eliminación
    if "call_id" in df_eliminated.columns:
        ids_raw = set(df_raw["call_id"].to_list())
        ids_clean = set(df_clean["call_id"].to_list())
        ids_eliminated = ids_raw - ids_clean
        df_elim_categorized = df_eliminated.filter(pl.col("call_id").is_in(list(ids_eliminated)))
    else:
        df_elim_categorized = df_eliminated

    def categorize(row):
        if row.get("hour") == 0:
            return "hora_0"
        if row.get("campaign_type") == "prueba":
            return "prueba"
        dur = row.get("duration_sec")
        if row.get("connected") == True and dur and dur > 600:
            return "outlier_dur"
        return "desconectadas"

    df_dict = df_elim_categorized.to_dicts()
    categories = [categorize(row) for row in df_dict]
    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Orden personalizado
    order = ["desconectadas", "outlier_dur", "hora_0", "prueba"]
    cats = [c for c in order if c in cat_counts]
    sizes = [cat_counts[c] for c in cats]
    colors = ["#4ecdc4", "#ff6b6b", "#45b7d1", "#ffd93d"][:len(cats)]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(cats, sizes, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Categoría de eliminación", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cantidad de registros", fontsize=12, fontweight="bold")
    ax.set_title(f"Categorización de {sum(sizes):,} registros eliminados", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Añadir etiquetas de cantidad en las barras
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        pct = size / sum(sizes) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{size:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sc3_elimination_categories.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico: sc3_elimination_categories.png")


if __name__ == "__main__":
    print("[INFO] Cargando datasets...")
    df_raw = pl.read_csv(RAW_CSV)
    df_clean = pl.read_csv(CLEAN_CSV)
    print(f"[INFO] Dataset original: {df_raw.height:,} registros")
    print(f"[INFO] Dataset limpio: {df_clean.height:,} registros")
    print()

    run(df_raw, df_clean)
