"""
Script 03 - Análisis de contactabilidad (Tarea 0).
Entrada : pl.DataFrame limpio (salida de clean_data.py)
Salida  : pl.DataFrame enriquecido con campaign_type + figuras en reports/figures/
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

from utils.paths import CLEAN_CSV, FIGURES_DIR
from utils.plotting import savefig, DPI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
FIGSIZE_BAR = (10, 5)
FIGSIZE_HEATMAP = (16, 4)
FIGSIZE_SCORE = (10, 6)
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK = 9
CACHE_MIN_KB = 5
MIN_WINDOW_CALLS = 10
MIN_HEATMAP_CALLS = 5

DOW_ORDER = ["do", "lu", "ma", "mi", "ju", "vi", "sa"]

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

# Rutas de todas las figuras generadas
_FIGURE_NAMES = [
    "contactability_by_hour", "contactability_by_dow",
    "contactability_by_campaign", "contactability_heatmap",
    "boxplot_duration_by_dow", "boxplot_duration_by_campaign",
    "heatmap_duration_hour_dow",
    "stacked_bar_reason_by_dow", "stacked_bar_reason_by_campaign",
    "heatmap_agent_hangup_hour_dow",
    "stacked_bar_sentiment_by_dow", "stacked_bar_sentiment_by_campaign",
    "heatmap_positive_sentiment_hour_dow",
    "optimal_windows_score",
]


def _fig_path(name: str) -> Path:
    return FIGURES_DIR / f"{name}.png"


# ---------------------------------------------------------------------------
# Helpers de visualización
# ---------------------------------------------------------------------------

def _bar_chart(x, y, xlabel, ylabel, title, out_path, color="steelblue", rotation=0):
    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    bars = ax.bar(x, y, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
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
            ha="center", va="bottom", fontsize=FONT_TICK,
        )
    savefig(fig, out_path)


def _heatmap(matrix, row_labels, col_labels, title, out_path,
             cmap="YlOrRd", val_fmt=None, colorbar_label="Tasa de conexión"):
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FONT_TICK)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Hora del día", fontsize=10)
    ax.set_ylabel("Día", fontsize=10)
    fmt = val_fmt if val_fmt else (lambda v: f"{v:.0%}")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, fmt(val), ha="center", va="center", fontsize=7, color=text_color)
    plt.colorbar(im, ax=ax, label=colorbar_label)
    savefig(fig, out_path)


def _boxplot_duration(df: pl.DataFrame, group_col: str, out_path: Path,
                      title: str, rotation: int = 0):
    """Boxplot de duration_sec por group_col (solo conectadas)."""
    connected = df.filter(pl.col("connected") == True)
    groups = _resolve_group_order(connected, group_col)

    data = [
        connected.filter(pl.col(group_col) == g)["duration_sec"].drop_nulls().to_list()
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    ax.boxplot(
        data, labels=groups, patch_artist=True,
        boxprops=dict(facecolor="#2196F333", color="#2196F3"),
        medianprops=dict(color="#F44336", linewidth=2),
        whiskerprops=dict(color="#555"),
        capprops=dict(color="#555"),
        flierprops=dict(marker=".", markersize=3, alpha=0.3, color="#aaa"),
    )
    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=FONT_LABEL)
    ax.set_ylabel("Duración (segundos)", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(bottom=0)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    savefig(fig, out_path)


def _stacked_bar(df: pl.DataFrame, group_col: str, cat_col: str,
                 cat_order: list[str], cat_colors: dict[str, str],
                 out_path: Path, title: str, rotation: int = 0):
    """Stacked bar 100% genérico por group_col (solo conectadas)."""
    connected = df.filter(pl.col("connected") == True)
    counts = connected.group_by([group_col, cat_col]).agg(pl.len().alias("n"))
    totals = connected.group_by(group_col).agg(pl.len().alias("total"))
    groups = _resolve_group_order(connected, group_col)

    # Determinar categorías presentes y ordenar
    present_cats = set(counts[cat_col].drop_nulls().to_list())
    ordered = [c for c in cat_order if c in present_cats]
    ordered += [c for c in present_cats if c not in ordered]

    merged = counts.join(totals, on=group_col).with_columns(
        (pl.col("n") / pl.col("total")).alias("pct")
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    bottom = np.zeros(len(groups))
    for cat in ordered:
        vals = np.array([
            merged.filter(
                (pl.col(group_col) == g) & (pl.col(cat_col) == cat)
            )["pct"].sum()
            for g in groups
        ])
        ax.bar(groups, vals, bottom=bottom, label=cat.replace("_", " "),
               color=cat_colors.get(cat, "#9E9E9E"), edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=FONT_LABEL)
    ax.set_ylabel("Proporción", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    savefig(fig, out_path)


def _optimal_windows_chart(df: pl.DataFrame, out_path: Path):
    """Top 10 ventanas hora x día por score compuesto (conexión x dur_norm x %positivo)."""
    base = (
        df.group_by(["day_of_week", "hour"])
        .agg([
            pl.len().alias("total"),
            pl.col("connected").cast(pl.Int32).sum().alias("n_conn"),
        ])
        .with_columns((pl.col("n_conn") / pl.col("total")).alias("conn_rate"))
        .filter(pl.col("total") >= MIN_WINDOW_CALLS, pl.col("day_of_week").is_in(DOW_ORDER))
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

    fig, ax = plt.subplots(figsize=FIGSIZE_SCORE)
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
    ax.set_title("Top 10 Ventanas Óptimas — Score Compuesto", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlim(0, x_max * 1.5)
    savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Helpers de datos
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


def _resolve_group_order(connected: pl.DataFrame, group_col: str) -> list:
    """Determina el orden de grupos: DOW_ORDER para días, por mediana desc para otros."""
    present = set(connected[group_col].unique().to_list())
    if group_col == "day_of_week":
        return [d for d in DOW_ORDER if d in present]
    medians = {
        g: (connected.filter(pl.col(group_col) == g)["duration_sec"].median() or 0)
        for g in present
    }
    return sorted(medians, key=lambda g: medians[g], reverse=True)


def _build_dow_hour_matrix(agg_df: pl.DataFrame, hours_all: list) -> np.ndarray:
    """Construye matriz 7 x N_hours desde agg_df con columnas day_of_week, hour, val."""
    hour_idx = {h: j for j, h in enumerate(hours_all)}
    dow_idx = {d: i for i, d in enumerate(DOW_ORDER)}
    matrix = np.zeros((len(DOW_ORDER), len(hours_all)))
    for row in agg_df.iter_rows(named=True):
        i = dow_idx.get(row["day_of_week"])
        j = hour_idx.get(row["hour"])
        if i is not None and j is not None:
            matrix[i, j] = row["val"]
    return matrix


def _rate_by_dow_hour(df: pl.DataFrame, filter_expr, num_expr,
                      min_total: int = 0) -> pl.DataFrame:
    """Agrega tasa (num_expr / total) por day_of_week x hour, con filtro opcional."""
    base = df.filter(filter_expr) if filter_expr is not None else df
    agg = (
        base.group_by(["day_of_week", "hour"])
        .agg([
            pl.len().alias("total"),
            num_expr.alias("n"),
        ])
        .with_columns((pl.col("n") / pl.col("total")).alias("val"))
    )
    if min_total > 0:
        agg = agg.filter(pl.col("total") >= min_total)
    return agg


# ---------------------------------------------------------------------------
# Generación de figuras por grupo
# ---------------------------------------------------------------------------

def _generate_connection_charts(df: pl.DataFrame, hours_all: list) -> pl.DataFrame:
    """Grupo original: tasa de conexión por hora, día, campaña y heatmap."""
    by_hour = _connection_rate(df, "hour").sort("hour")
    _bar_chart(
        by_hour["hour"].to_list(), by_hour["connection_rate"].to_list(),
        "Hora del día", "Tasa de conexión",
        "Tasa de Conexión por Hora del Día",
        _fig_path("contactability_by_hour"), color="#2196F3",
    )

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
        _fig_path("contactability_by_dow"), color="#4CAF50",
    )

    by_camp = _connection_rate(df, "campaign_type").sort("connection_rate", descending=True)
    _bar_chart(
        by_camp["campaign_type"].to_list(), by_camp["connection_rate"].to_list(),
        "Tipo de campaña", "Tasa de conexión",
        "Tasa de Conexión por Tipo de Campaña",
        _fig_path("contactability_by_campaign"), color="#FF9800", rotation=30,
    )

    rates_agg = _rate_by_dow_hour(
        df.with_columns(pl.col("connected").cast(pl.Int32)),
        filter_expr=None,
        num_expr=pl.col("connected").sum(),
    )
    matrix_conn = _build_dow_hour_matrix(rates_agg, hours_all)
    _heatmap(
        matrix_conn, DOW_ORDER, hours_all,
        "Tasa de Conexión: Hora × Día de la Semana",
        _fig_path("contactability_heatmap"),
    )

    return rates_agg


def _generate_duration_charts(df: pl.DataFrame, hours_all: list) -> None:
    """Grupo A: duración por día, campaña y heatmap."""
    _boxplot_duration(
        df, "day_of_week", _fig_path("boxplot_duration_by_dow"),
        "Duración de Llamada por Día de la Semana",
    )
    _boxplot_duration(
        df, "campaign_type", _fig_path("boxplot_duration_by_campaign"),
        "Duración de Llamada por Tipo de Campaña", rotation=30,
    )
    dur_agg = (
        df.filter(pl.col("connected") == True)
        .group_by(["day_of_week", "hour"])
        .agg(pl.col("duration_sec").mean().alias("val"))
    )
    _heatmap(
        _build_dow_hour_matrix(dur_agg, hours_all), DOW_ORDER, hours_all,
        "Duración Promedio de Llamada: Hora × Día (segundos)",
        _fig_path("heatmap_duration_hour_dow"),
        cmap="Blues", val_fmt=lambda v: f"{v:.0f}s", colorbar_label="Duración media (s)",
    )


def _generate_reason_charts(df: pl.DataFrame, hours_all: list) -> None:
    """Grupo B: razón de desconexión por día, campaña y heatmap agent_hangup."""
    _stacked_bar(
        df, "day_of_week", "disconnected_reason", _REASON_ORDER, _REASON_COLORS,
        _fig_path("stacked_bar_reason_by_dow"),
        "Razón de Desconexión por Día de la Semana",
    )
    _stacked_bar(
        df, "campaign_type", "disconnected_reason", _REASON_ORDER, _REASON_COLORS,
        _fig_path("stacked_bar_reason_by_campaign"),
        "Razón de Desconexión por Tipo de Campaña", rotation=30,
    )
    hangup_agg = _rate_by_dow_hour(
        df, filter_expr=pl.col("connected") == True,
        num_expr=(pl.col("disconnected_reason") == "agent_hangup").cast(pl.Int32).sum(),
        min_total=MIN_HEATMAP_CALLS,
    )
    _heatmap(
        _build_dow_hour_matrix(hangup_agg, hours_all), DOW_ORDER, hours_all,
        "Tasa de Agent Hangup (llamada completada): Hora × Día",
        _fig_path("heatmap_agent_hangup_hour_dow"),
        cmap="Greens", colorbar_label="Tasa agent_hangup",
    )


def _generate_sentiment_charts(df: pl.DataFrame, hours_all: list) -> None:
    """Grupo C: sentimiento por día, campaña y heatmap positivo."""
    _stacked_bar(
        df, "day_of_week", "sentiment_own", _SENTIMENT_ORDER, _SENTIMENT_COLORS,
        _fig_path("stacked_bar_sentiment_by_dow"),
        "Distribución de Sentimiento por Día de la Semana",
    )
    _stacked_bar(
        df, "campaign_type", "sentiment_own", _SENTIMENT_ORDER, _SENTIMENT_COLORS,
        _fig_path("stacked_bar_sentiment_by_campaign"),
        "Distribución de Sentimiento por Tipo de Campaña", rotation=30,
    )
    sent_agg = _rate_by_dow_hour(
        df, filter_expr=pl.col("connected") == True,
        num_expr=(pl.col("sentiment_own") == "positivo").cast(pl.Int32).sum(),
        min_total=MIN_HEATMAP_CALLS,
    )
    _heatmap(
        _build_dow_hour_matrix(sent_agg, hours_all), DOW_ORDER, hours_all,
        "Tasa de Sentimiento Positivo: Hora × Día",
        _fig_path("heatmap_positive_sentiment_hour_dow"),
        cmap="RdYlGn", colorbar_label="% sentimiento positivo",
    )


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------

def _figures_cached() -> bool:
    """Verifica si todas las figuras existen y tienen tamaño mínimo."""
    paths = [_fig_path(name) for name in _FIGURE_NAMES]
    if all(p.exists() and p.stat().st_size > CACHE_MIN_KB * 1024 for p in paths):
        print("[INFO] Cache encontrado: figuras de contactabilidad ya existen")
        return True
    if any(p.exists() for p in paths):
        print("[WARN] Figuras corruptas o incompletas, regenerando...")
    return False


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def analyze_contactability(df: pl.DataFrame) -> pl.DataFrame:
    """Analiza patrones de contactabilidad y genera visualizaciones."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _add_campaign_type(df)

    if _figures_cached():
        return df

    hours_all = sorted(df["hour"].unique().to_list())

    rates_agg = _generate_connection_charts(df, hours_all)
    _generate_duration_charts(df, hours_all)
    _generate_reason_charts(df, hours_all)
    _generate_sentiment_charts(df, hours_all)
    _optimal_windows_chart(df, _fig_path("optimal_windows_score"))

    # Log ventana óptima
    top = (
        rates_agg
        .filter(pl.col("total") >= MIN_WINDOW_CALLS, pl.col("day_of_week").is_in(DOW_ORDER))
        .sort("val", descending=True)
        .head(1)
        .to_dicts()
    )
    print(f"[INFO] Figuras de contactabilidad guardadas en: {FIGURES_DIR}")
    if top:
        best = top[0]
        print(f"[INFO] Ventana optima: {best['day_of_week']} {best['hour']}h -> {best['val']:.1%} conexion")

    return df


if __name__ == "__main__":
    df = pl.read_csv(CLEAN_CSV)
    analyze_contactability(df)
