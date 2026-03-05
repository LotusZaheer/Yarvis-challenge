"""
Script 03 - Análisis de contactabilidad (Tarea 0).
Entrada : pl.DataFrame limpio (salida de clean_data.py)
Salida  : pl.DataFrame enriquecido con campaign_type + figuras en reports/figures/
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "calls_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

DOW_ORDER = ["lu", "ma", "mi", "ju", "vi", "sa", "do"]

# Patrones regex para clasificar el campo `name` en tipo de campaña
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


def _classify_campaign(name: str) -> str:
    if not name:
        return "otro"
    n = name.lower()
    for pattern, label in _CAMPAIGN_PATTERNS:
        if re.search(pattern, n):
            return label
    return "otro"


def _add_campaign_type(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("name")
        .map_elements(lambda n: _classify_campaign(n or ""), return_dtype=pl.Utf8)
        .alias("campaign_type")
    )


def _connection_rate(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Calcula tasa de conexión agrupando por group_col."""
    return (
        df.group_by(group_col)
        .agg([
            pl.len().alias("total"),
            pl.col("connected").cast(pl.Int32).sum().alias("connected_count"),
        ])
        .with_columns(
            (pl.col("connected_count") / pl.col("total")).alias("connection_rate")
        )
    )


def _bar_chart(x, y, xlabel, ylabel, title, out_path, color="steelblue", rotation=0):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, y, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    y_max = max(y) if any(v > 0 for v in y) else 0.5
    ax.set_ylim(0, y_max * 1.25)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.01,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _heatmap(matrix, row_labels, col_labels, title, out_path):
    fig, ax = plt.subplots(figsize=(16, 4))
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Hora del día", fontsize=10)
    ax.set_ylabel("Día", fontsize=10)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7, color=text_color)
    plt.colorbar(im, ax=ax, label="Tasa de conexión")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_contactability(df: pl.DataFrame) -> pl.DataFrame:
    """
    Analiza patrones de contactabilidad y genera visualizaciones.
    Si las 4 figuras ya existen, omite la generación y solo agrega campaign_type.
    Retorna df enriquecido con columna campaign_type.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _add_campaign_type(df)

    _figures = [
        FIGURES_DIR / "contactability_by_hour.png",
        FIGURES_DIR / "contactability_by_dow.png",
        FIGURES_DIR / "contactability_by_campaign.png",
        FIGURES_DIR / "contactability_heatmap.png",
    ]
    # Validar que todas las figuras existen y tienen tamaño mínimo (5KB) para detectar corrupción
    _MIN_SIZE_KB = 5
    if all(f.exists() and f.stat().st_size > _MIN_SIZE_KB * 1024 for f in _figures):
        print("[INFO] Cache encontrado: figuras de contactabilidad ya existen")
        return df
    if any(f.exists() for f in _figures):
        print(f"[WARN] Figuras corruptas o incompletas, regenerando...")

    # --- Tasa por hora ---
    by_hour = _connection_rate(df, "hour").sort("hour")
    _bar_chart(
        by_hour["hour"].to_list(),
        by_hour["connection_rate"].to_list(),
        "Hora del día", "Tasa de conexión",
        "Tasa de Conexión por Hora del Día",
        FIGURES_DIR / "contactability_by_hour.png",
        color="#2196F3",
    )

    # --- Tasa por día de la semana ---
    by_dow = _connection_rate(df, "day_of_week")
    by_dow_sorted = (
        by_dow
        .with_columns(
            pl.col("day_of_week")
            .replace(DOW_ORDER, list(range(len(DOW_ORDER))), default=99)
            .cast(pl.Int8)
            .alias("_rank")
        )
        .sort("_rank")
        .drop("_rank")
    )
    _bar_chart(
        by_dow_sorted["day_of_week"].to_list(),
        by_dow_sorted["connection_rate"].to_list(),
        "Día de la semana", "Tasa de conexión",
        "Tasa de Conexión por Día de la Semana",
        FIGURES_DIR / "contactability_by_dow.png",
        color="#4CAF50",
    )

    # --- Tasa por tipo de campaña ---
    by_camp = _connection_rate(df, "campaign_type").sort("connection_rate", descending=True)
    _bar_chart(
        by_camp["campaign_type"].to_list(),
        by_camp["connection_rate"].to_list(),
        "Tipo de campaña", "Tasa de conexión",
        "Tasa de Conexión por Tipo de Campaña",
        FIGURES_DIR / "contactability_by_campaign.png",
        color="#FF9800", rotation=30,
    )

    # --- Heatmap hora × día ---
    # Calcular tasas con group_by vectorizado (un solo pase sobre el df completo)
    rates_df = (
        df.with_columns(pl.col("connected").cast(pl.Int32))
        .group_by(["day_of_week", "hour"])
        .agg([
            pl.len().alias("total"),
            pl.col("connected").sum().alias("connected_count"),
        ])
        .with_columns((pl.col("connected_count") / pl.col("total")).alias("rate"))
    )

    hours_all = sorted(df["hour"].unique().to_list())
    dows_in = [d for d in DOW_ORDER if d in df["day_of_week"].unique().to_list()]
    hour_idx = {h: j for j, h in enumerate(hours_all)}
    dow_idx = {d: i for i, d in enumerate(dows_in)}

    # Poblar matrix iterando sobre el df ya agregado (≤168 filas, no 73k)
    matrix = np.zeros((len(dows_in), len(hours_all)))
    for row in rates_df.filter(pl.col("day_of_week").is_in(dows_in)).iter_rows(named=True):
        i = dow_idx.get(row["day_of_week"])
        j = hour_idx.get(row["hour"])
        if i is not None and j is not None:
            matrix[i, j] = row["rate"]

    _heatmap(
        matrix, dows_in, hours_all,
        "Tasa de Conexión: Hora × Día de la Semana",
        FIGURES_DIR / "contactability_heatmap.png",
    )

    # Ventana óptima: top 5 combos hora×día con >= 10 llamadas (vectorizado)
    top_window = (
        rates_df
        .filter(pl.col("total") >= 10, pl.col("day_of_week").is_in(dows_in))
        .rename({"day_of_week": "day"})
        .sort("rate", descending=True)
        .head(5)
        .to_dicts()
    )

    print(f"[INFO] Figuras de contactabilidad guardadas en: {FIGURES_DIR}")
    if top_window:
        best = top_window[0]
        print(f"[INFO] Ventana optima: {best['day']} {best['hour']}h -> {best['rate']:.1%} conexion")

    return df


if __name__ == "__main__":
    df = pl.read_csv(CLEAN_CSV)
    analyze_contactability(df)
