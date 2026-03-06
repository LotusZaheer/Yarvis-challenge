"""
Script 06 - Análisis de desempeño del agente Yarvis (Tarea 3).
Entrada : pl.DataFrame con sentiment_own y cluster_id
Salida  : pl.DataFrame con columnas de flags de desempeño + figura en reports/figures/
          (no exporta CSV separado; los hallazgos van al reporte)
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import polars as pl

from utils.cache import load_csv_cache
from utils.df_helpers import connected_calls
from utils.paths import FIGURES_DIR, PROCESSED_DIR, ensure_output_dirs
from utils.plotting import savefig
from utils.text import extract_transcript_lines, normalize_text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AGENT_CACHE = PROCESSED_DIR / "cache_agent.csv"

# Umbral de duración corta (segundos) para clasificar llamadas tipo "malentendido"
SHORT_CALL_THRESHOLD_SEC = 30.0

def _has_repetitive_responses(transcript: str, min_repeat: int = 2) -> bool:
    """Detecta si el agente repite la misma frase >= min_repeat veces."""
    lines = extract_transcript_lines(transcript or "", "Agent")
    if len(lines) < min_repeat:
        return False
    normalized = [normalize_text(l) for l in lines if len(l) > 20]
    counts = Counter(normalized)
    return any(c >= min_repeat for c in counts.values())


def _detect_failures(df: pl.DataFrame) -> pl.DataFrame:
    """
    Añade 5 columnas booleanas de fallo usando operaciones vectorizadas.
    Solo aplica sobre llamadas conectadas; el resto recibe False.
    """
    connected = pl.col("connected") == True
    sent = pl.col("sentiment_own") if "sentiment_own" in df.columns else pl.lit("neutral")

    return df.with_columns([
        # 1. Respuestas repetitivas: map_elements solo sobre texto (necesita lógica Python por llamada)
        pl.when(connected)
        .then(
            pl.col("transcript_text")
            .map_elements(
                lambda t: _has_repetitive_responses(t or ""), return_dtype=pl.Boolean
            )
        )
        .otherwise(False)
        .alias("fail_repetitive"),

        # 2. Inactividad: condición escalar vectorizable
        pl.when(connected)
        .then(pl.col("disconnected_reason") == "inactivity")
        .otherwise(False)
        .alias("fail_inactivity"),

        # 3. No manejo de objeciones: multi-columna con when/then
        pl.when(connected)
        .then(
            pl.col("pca_razon_churn").is_not_null()
            & (sent == "negativo")
            & (
                pl.col("pca_posible_recuperacion").is_null()
                | pl.col("pca_posible_recuperacion").str.to_lowercase().is_in(["no"])
            )
        )
        .otherwise(False)
        .alias("fail_objection"),

        # 4. Malentendido: llamada corta + usuario cuelga + sentimiento negativo
        pl.when(connected)
        .then(
            (pl.col("duration_sec").fill_null(0.0) < SHORT_CALL_THRESHOLD_SEC)
            & (pl.col("disconnected_reason") == "user_hangup")
            & (sent == "negativo")
        )
        .otherwise(False)
        .alias("fail_misunderstanding"),

        # 5. Agente cuelga activamente: señal directa de abandono o confusión del agente
        pl.when(connected)
        .then(pl.col("disconnected_reason") == "agent_hangup")
        .otherwise(False)
        .alias("fail_agent_hangup"),
    ])


def _print_summary(df: pl.DataFrame) -> None:
    """Imprime resumen de fallas detectadas."""
    connected = connected_calls(df)
    n = connected.height
    if n == 0:
        print("[WARN] No hay llamadas conectadas para analizar desempeño.")
        return

    failures = {
        "Respuestas repetitivas":    connected["fail_repetitive"].sum(),
        "Inactividad (timeout)":     connected["fail_inactivity"].sum(),
        "Sin manejo de objecion":    connected["fail_objection"].sum(),
        "Malentendido/llamada corta": connected["fail_misunderstanding"].sum(),
        "Agente cuelga (hangup)":    connected["fail_agent_hangup"].sum(),
    }
    print(f"[INFO] Desempeno del agente ({n:,} llamadas conectadas):")
    for label, count in failures.items():
        print(f"       {label}: {count:,} ({count/n:.1%})")


def _plot_failures(df: pl.DataFrame, out_path: Path):
    """Gráfico de barras con frecuencia de cada tipo de falla."""
    connected = connected_calls(df)
    n = connected.height
    labels = [
        "Respuestas\nrepetitivas",
        "Inactividad\n(timeout)",
        "Sin manejo\nobjeción",
        "Malentendido\nllamada corta",
        "Agente\ncuelga",
    ]
    cols = ["fail_repetitive", "fail_inactivity", "fail_objection", "fail_misunderstanding", "fail_agent_hangup"]
    counts = [connected[c].sum() for c in cols]
    pcts = [c / n for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, pcts, color=["#F44336", "#FF9800", "#9C27B0", "#2196F3", "#607D8B"], edgecolor="white")
    ax.set_ylabel("% de llamadas conectadas")
    ax.set_title("Patrones de Falla del Agente Yarvis", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    for bar, pct, count in zip(bars, pcts, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{pct:.1%}\n({count:,})",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, max(pcts) * 1.3 if any(p > 0 for p in pcts) else 0.1)
    savefig(fig, out_path)


def analyze_agent_performance(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detecta patrones de mal desempeño del agente.
    Si cache_agent.csv ya existe, carga las columnas fail_* desde ahí (join por call_url).
    Retorna df con columnas fail_* añadidas.
    """
    ensure_output_dirs()
    _fail_cols = ["fail_repetitive", "fail_inactivity", "fail_objection", "fail_misunderstanding", "fail_agent_hangup"]
    _figure = FIGURES_DIR / "agent_failures.png"

    cached = load_csv_cache(AGENT_CACHE, expected_rows=df.height, expected_cols=_fail_cols)
    if cached is not None:
        df = df.join(cached.select(["call_url"] + _fail_cols), on="call_url", how="left")
        for col in _fail_cols:
            df = df.with_columns(pl.col(col).fill_null(False))
        if not _figure.exists() or _figure.stat().st_size < 2 * 1024:
            _plot_failures(df, _figure)
        return df

    df = _detect_failures(df)
    _print_summary(df)
    _plot_failures(df, _figure)
    print(f"[INFO] Figura de desempeno guardada en: {_figure}")

    # Guardar checkpoint (call_url + fail_*)
    df.select(["call_url"] + _fail_cols).write_csv(AGENT_CACHE)

    return df


if __name__ == "__main__":
    from utils.paths import CLEAN_CSV
    from scripts.sentiment_analysis import analyze_sentiment  # noqa: E402

    df = pl.read_csv(CLEAN_CSV)
    df = analyze_sentiment(df)
    df = analyze_agent_performance(df)
    print(df.filter(pl.col("connected") == True)
            .select(["target_id", "sentiment_own", "fail_repetitive", "fail_inactivity",
                     "fail_objection", "fail_misunderstanding"])
            .head(10))
