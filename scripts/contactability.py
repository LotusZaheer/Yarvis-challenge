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

DOW_ORDER = [ "do", "lu", "ma", "mi", "ju", "vi", "sa"]

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

_REASON_ORDER = [
    "agent_hangup", "user_hangup", "inactivity",
    "ivr_reached", "max_duration_reached", "system_error",
]
_REASON_COLORS = {
    "agent_hangup": "#4CAF50", "user_hangup": "#2196F3", "inactivity": "#FF9800",
    "ivr_reached": "#9C27B0", "max_duration_reached": "#F44336", "system_error": "#607D8B",
}

_SENTIMENT_ORDER = ["positivo", "neutral", "negativo"]
_SENTIMENT_COLORS = {"positivo": "#4CAF50", "neutral": "#9E9E9E", "negativo": "#F44336"}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

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


def _heatmap(matrix, row_labels, col_labels, title, out_path,
             cmap="YlOrRd", val_fmt=None, colorbar_label="Tasa de conexión"):
    fig, ax = plt.subplots(figsize=(16, 4))
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Hora del día", fontsize=10)
    ax.set_ylabel("Día", fontsize=10)
    fmt = val_fmt if val_fmt else (lambda v: f"{v:.0%}")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, fmt(val), ha="center", va="center", fontsize=7, color=text_color)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_dow_hour_matrix(agg_df: pl.DataFrame, hours_all: list) -> np.ndarray:
    """Construye matriz 7×N_hours desde agg_df con columnas day_of_week, hour, val."""
    hour_idx = {h: j for j, h in enumerate(hours_all)}
    dow_idx = {d: i for i, d in enumerate(DOW_ORDER)}
    matrix = np.zeros((len(DOW_ORDER), len(hours_all)))
    for row in agg_df.iter_rows(named=True):
        i = dow_idx.get(row["day_of_week"])
        j = hour_idx.get(row["hour"])
        if i is not None and j is not None:
            matrix[i, j] = row["val"]
    return matrix


def _boxplot_duration(df: pl.DataFrame, group_col: str, out_path: Path,
                      title: str, rotation: int = 0):
    """Boxplot de duration_sec por group_col (solo conectadas)."""
    connected = df.filter(pl.col("connected") == True)
    present = set(connected[group_col].unique().to_list())

    if group_col == "day_of_week":
        groups = [d for d in DOW_ORDER if d in present]
    else:
        medians = {
            g: (connected.filter(pl.col(group_col) == g)["duration_sec"].median() or 0)
            for g in present
        }
        groups = sorted(medians, key=lambda g: medians[g], reverse=True)

    data = [
        connected.filter(pl.col(group_col) == g)["duration_sec"].drop_nulls().to_list()
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        data, labels=groups, patch_artist=True,
        boxprops=dict(facecolor="#2196F333", color="#2196F3"),
        medianprops=dict(color="#F44336", linewidth=2),
        whiskerprops=dict(color="#555"),
        capprops=dict(color="#555"),
        flierprops=dict(marker=".", markersize=3, alpha=0.3, color="#aaa"),
    )
    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel("Duración (segundos)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _stacked_bar_reason(df: pl.DataFrame, group_col: str, out_path: Path,
                        title: str, rotation: int = 0):
    """Stacked bar 100% de disconnected_reason por group_col (solo conectadas)."""
    connected = df.filter(pl.col("connected") == True)
    counts = (
        connected.group_by([group_col, "disconnected_reason"])
        .agg(pl.len().alias("n"))
    )
    totals = connected.group_by(group_col).agg(pl.len().alias("total"))

    if group_col == "day_of_week":
        present = set(connected[group_col].unique().to_list())
        groups = [d for d in DOW_ORDER if d in present]
    else:
        groups = totals.sort("total", descending=True)[group_col].to_list()

    present_reasons = set(counts["disconnected_reason"].drop_nulls().to_list())
    cat_order = [r for r in _REASON_ORDER if r in present_reasons]
    cat_order += [r for r in present_reasons if r not in cat_order]

    merged = counts.join(totals, on=group_col).with_columns(
        (pl.col("n") / pl.col("total")).alias("pct")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(groups))
    for cat in cat_order:
        vals = np.array([
            merged.filter(
                (pl.col(group_col) == g) & (pl.col("disconnected_reason") == cat)
            )["pct"].sum()
            for g in groups
        ])
        ax.bar(groups, vals, bottom=bottom, label=cat.replace("_", " "),
               color=_REASON_COLORS.get(cat, "#9E9E9E"), edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel("Proporción", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _stacked_bar_sentiment(df: pl.DataFrame, group_col: str, out_path: Path,
                           title: str, rotation: int = 0):
    """Stacked bar 100% de sentiment_own por group_col (solo conectadas)."""
    connected = df.filter(pl.col("connected") == True)
    counts = (
        connected.group_by([group_col, "sentiment_own"])
        .agg(pl.len().alias("n"))
    )
    totals = connected.group_by(group_col).agg(pl.len().alias("total"))

    if group_col == "day_of_week":
        present = set(connected[group_col].unique().to_list())
        groups = [d for d in DOW_ORDER if d in present]
    else:
        groups = totals.sort("total", descending=True)[group_col].to_list()

    merged = counts.join(totals, on=group_col).with_columns(
        (pl.col("n") / pl.col("total")).alias("pct")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(groups))
    for sent in _SENTIMENT_ORDER:
        vals = np.array([
            merged.filter(
                (pl.col(group_col) == g) & (pl.col("sentiment_own") == sent)
            )["pct"].sum()
            for g in groups
        ])
        ax.bar(groups, vals, bottom=bottom, label=sent,
               color=_SENTIMENT_COLORS[sent], edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel("Proporción", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right")
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _optimal_windows_chart(df: pl.DataFrame, out_path: Path):
    """Top 10 ventanas hora×día por score compuesto (conexión × dur_norm × %positivo)."""
    base = (
        df.group_by(["day_of_week", "hour"])
        .agg([
            pl.len().alias("total"),
            pl.col("connected").cast(pl.Int32).sum().alias("n_conn"),
        ])
        .with_columns((pl.col("n_conn") / pl.col("total")).alias("conn_rate"))
        .filter(pl.col("total") >= 10, pl.col("day_of_week").is_in(DOW_ORDER))
    )

    connected = df.filter(pl.col("connected") == True)
    quality = (
        connected.group_by(["day_of_week", "hour"])
        .agg([
            pl.col("duration_sec").mean().alias("dur_mean"),
            (
                (pl.col("sentiment_own") == "positivo").cast(pl.Int32).sum() / pl.len()
            ).alias("pct_pos"),
        ])
    )

    combined = base.join(quality, on=["day_of_week", "hour"], how="left").fill_null(0)

    dur_max = combined["dur_mean"].max()
    dur_min = combined["dur_mean"].min()
    dur_expr = (
        (pl.col("dur_mean") - dur_min) / (dur_max - dur_min)
        if dur_max > dur_min
        else pl.lit(0.0)
    )
    combined = (
        combined
        .with_columns(dur_expr.alias("dur_norm"))
        .with_columns(
            (pl.col("conn_rate") * pl.col("dur_norm") * pl.col("pct_pos")).alias("score")
        )
        .sort("score", descending=True)
        .head(10)
    )

    labels = [f"{r['day_of_week']} {r['hour']}h" for r in combined.iter_rows(named=True)]
    scores = combined["score"].to_list()
    conn_rates = combined["conn_rate"].to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(labels)))
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor="white")

    x_max = max(scores) if scores else 1
    for bar, cr in zip(bars, conn_rates[::-1]):
        ax.text(
            bar.get_width() + x_max * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"conexión: {cr:.1%}",
            va="center", fontsize=8, color="#555",
        )

    ax.set_xlabel("Score compuesto (conexión × dur_norm × %positivo)", fontsize=10)
    ax.set_title("Top 10 Ventanas Óptimas — Score Compuesto", fontsize=13, fontweight="bold")
    ax.set_xlim(0, x_max * 1.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def analyze_contactability(df: pl.DataFrame) -> pl.DataFrame:
    """
    Analiza patrones de contactabilidad y genera visualizaciones.
    Retorna df enriquecido con columna campaign_type.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _add_campaign_type(df)

    _figures = [
        # Originales (tasa de conexión)
        FIGURES_DIR / "contactability_by_hour.png",
        FIGURES_DIR / "contactability_by_dow.png",
        FIGURES_DIR / "contactability_by_campaign.png",
        FIGURES_DIR / "contactability_heatmap.png",
        # Grupo A — Duración
        FIGURES_DIR / "boxplot_duration_by_dow.png",
        FIGURES_DIR / "boxplot_duration_by_campaign.png",
        FIGURES_DIR / "heatmap_duration_hour_dow.png",
        # Grupo B — Razón de desconexión
        FIGURES_DIR / "stacked_bar_reason_by_dow.png",
        FIGURES_DIR / "stacked_bar_reason_by_campaign.png",
        FIGURES_DIR / "heatmap_agent_hangup_hour_dow.png",
        # Grupo C — Sentimiento
        FIGURES_DIR / "stacked_bar_sentiment_by_dow.png",
        FIGURES_DIR / "stacked_bar_sentiment_by_campaign.png",
        FIGURES_DIR / "heatmap_positive_sentiment_hour_dow.png",
        # Grupo D — Score compuesto
        FIGURES_DIR / "optimal_windows_score.png",
    ]

    _MIN_SIZE_KB = 5
    if all(f.exists() and f.stat().st_size > _MIN_SIZE_KB * 1024 for f in _figures):
        print("[INFO] Cache encontrado: figuras de contactabilidad ya existen")
        return df
    if any(f.exists() for f in _figures):
        print("[WARN] Figuras corruptas o incompletas, regenerando...")

    hours_all = sorted(df["hour"].unique().to_list())

    # --- Tasa de conexión por hora ---
    by_hour = _connection_rate(df, "hour").sort("hour")
    _bar_chart(
        by_hour["hour"].to_list(), by_hour["connection_rate"].to_list(),
        "Hora del día", "Tasa de conexión",
        "Tasa de Conexión por Hora del Día",
        FIGURES_DIR / "contactability_by_hour.png", color="#2196F3",
    )

    # --- Tasa de conexión por día de la semana ---
    by_dow = _connection_rate(df, "day_of_week")
    by_dow_complete = pl.DataFrame({
        "day_of_week": DOW_ORDER,
        "total": [0] * len(DOW_ORDER),
        "connected_count": [0] * len(DOW_ORDER),
        "connection_rate": [0.0] * len(DOW_ORDER),
    }).update(by_dow, on="day_of_week")
    _bar_chart(
        by_dow_complete["day_of_week"].to_list(), by_dow_complete["connection_rate"].to_list(),
        "Día de la semana", "Tasa de conexión",
        "Tasa de Conexión por Día de la Semana",
        FIGURES_DIR / "contactability_by_dow.png", color="#4CAF50",
    )

    # --- Tasa de conexión por campaña ---
    by_camp = _connection_rate(df, "campaign_type").sort("connection_rate", descending=True)
    _bar_chart(
        by_camp["campaign_type"].to_list(), by_camp["connection_rate"].to_list(),
        "Tipo de campaña", "Tasa de conexión",
        "Tasa de Conexión por Tipo de Campaña",
        FIGURES_DIR / "contactability_by_campaign.png", color="#FF9800", rotation=30,
    )

    # --- Heatmap tasa de conexión hora × día ---
    rates_df = (
        df.with_columns(pl.col("connected").cast(pl.Int32))
        .group_by(["day_of_week", "hour"])
        .agg([pl.len().alias("total"), pl.col("connected").sum().alias("n_conn")])
        .with_columns((pl.col("n_conn") / pl.col("total")).alias("val"))
    )
    matrix_conn = _build_dow_hour_matrix(rates_df, hours_all)
    _heatmap(
        matrix_conn, DOW_ORDER, hours_all,
        "Tasa de Conexión: Hora × Día de la Semana",
        FIGURES_DIR / "contactability_heatmap.png",
    )

    # --- Grupo A: Duración ---
    _boxplot_duration(
        df, "day_of_week", FIGURES_DIR / "boxplot_duration_by_dow.png",
        "Duración de Llamada por Día de la Semana",
    )
    _boxplot_duration(
        df, "campaign_type", FIGURES_DIR / "boxplot_duration_by_campaign.png",
        "Duración de Llamada por Tipo de Campaña", rotation=30,
    )
    dur_agg = (
        df.filter(pl.col("connected") == True)
        .group_by(["day_of_week", "hour"])
        .agg(pl.col("duration_sec").mean().alias("val"))
    )
    matrix_dur = _build_dow_hour_matrix(dur_agg, hours_all)
    _heatmap(
        matrix_dur, DOW_ORDER, hours_all,
        "Duración Promedio de Llamada: Hora × Día (segundos)",
        FIGURES_DIR / "heatmap_duration_hour_dow.png",
        cmap="Blues",
        val_fmt=lambda v: f"{v:.0f}s",
        colorbar_label="Duración media (s)",
    )

    # --- Grupo B: Razón de desconexión ---
    _stacked_bar_reason(
        df, "day_of_week", FIGURES_DIR / "stacked_bar_reason_by_dow.png",
        "Razón de Desconexión por Día de la Semana",
    )
    _stacked_bar_reason(
        df, "campaign_type", FIGURES_DIR / "stacked_bar_reason_by_campaign.png",
        "Razón de Desconexión por Tipo de Campaña", rotation=30,
    )
    hangup_agg = (
        df.filter(pl.col("connected") == True)
        .group_by(["day_of_week", "hour"])
        .agg([
            pl.len().alias("total"),
            (pl.col("disconnected_reason") == "agent_hangup").cast(pl.Int32).sum().alias("n"),
        ])
        .filter(pl.col("total") >= 5)
        .with_columns((pl.col("n") / pl.col("total")).alias("val"))
    )
    matrix_hangup = _build_dow_hour_matrix(hangup_agg, hours_all)
    _heatmap(
        matrix_hangup, DOW_ORDER, hours_all,
        "Tasa de Agent Hangup (llamada completada): Hora × Día",
        FIGURES_DIR / "heatmap_agent_hangup_hour_dow.png",
        cmap="Greens",
        colorbar_label="Tasa agent_hangup",
    )

    # --- Grupo C: Sentimiento ---
    _stacked_bar_sentiment(
        df, "day_of_week", FIGURES_DIR / "stacked_bar_sentiment_by_dow.png",
        "Distribución de Sentimiento por Día de la Semana",
    )
    _stacked_bar_sentiment(
        df, "campaign_type", FIGURES_DIR / "stacked_bar_sentiment_by_campaign.png",
        "Distribución de Sentimiento por Tipo de Campaña", rotation=30,
    )
    sent_agg = (
        df.filter(pl.col("connected") == True)
        .group_by(["day_of_week", "hour"])
        .agg([
            pl.len().alias("total"),
            (pl.col("sentiment_own") == "positivo").cast(pl.Int32).sum().alias("n"),
        ])
        .filter(pl.col("total") >= 5)
        .with_columns((pl.col("n") / pl.col("total")).alias("val"))
    )
    matrix_sent = _build_dow_hour_matrix(sent_agg, hours_all)
    _heatmap(
        matrix_sent, DOW_ORDER, hours_all,
        "Tasa de Sentimiento Positivo: Hora × Día",
        FIGURES_DIR / "heatmap_positive_sentiment_hour_dow.png",
        cmap="RdYlGn",
        colorbar_label="% sentimiento positivo",
    )

    # --- Grupo D: Score compuesto ---
    _optimal_windows_chart(df, FIGURES_DIR / "optimal_windows_score.png")

    # Ventana óptima por conexión (log)
    top_window = (
        rates_df
        .filter(pl.col("total") >= 10, pl.col("day_of_week").is_in(DOW_ORDER))
        .rename({"day_of_week": "day"})
        .sort("val", descending=True)
        .head(5)
        .to_dicts()
    )
    print(f"[INFO] Figuras de contactabilidad guardadas en: {FIGURES_DIR}")
    if top_window:
        best = top_window[0]
        print(f"[INFO] Ventana optima: {best['day']} {best['hour']}h -> {best['val']:.1%} conexion")

    return df


if __name__ == "__main__":
    df = pl.read_csv(CLEAN_CSV)
    analyze_contactability(df)
