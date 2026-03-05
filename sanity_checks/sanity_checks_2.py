"""
Sanity Check 2 — Análisis de posibles valores no útiles.
Evalúa si las llamadas de hora 0 (00:00–00:59) y campaña "prueba"
son estadísticamente distintas del resto para justificar su exclusión.

Escribe resultados en: data/interim/sc2_outlier_analysis.txt
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "calls_clean.csv"
OUT_DIR   = PROJECT_ROOT / "data" / "interim"
OUT_FILE  = OUT_DIR / "sc2_outlier_analysis.txt"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proportions_ztest(n1, count1, n2, count2) -> tuple[float, float]:
    """Z-test de dos proporciones. Retorna (z_stat, p_value)."""
    p1 = count1 / n1 if n1 > 0 else 0
    p2 = count2 / n2 if n2 > 0 else 0
    p_pool = (count1 + count2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def _mannwhitney(a: list, b: list) -> tuple[float, float]:
    """Mann-Whitney U para comparar dos distribuciones de duración."""
    try:
        a = [float(x) for x in a if x is not None]
        b = [float(x) for x in b if x is not None]
    except (ValueError, TypeError):
        return float("nan"), float("nan")
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _describe(series: pl.Series) -> dict:
    # Filter to non-null numeric values
    filtered = series.drop_nulls()
    if filtered.is_empty():
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0}
    try:
        arr = filtered.cast(pl.Float64).to_numpy()
        arr = np.asarray(arr, dtype=float)
    except (ValueError, TypeError):
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0}
    if len(arr) == 0:
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0}
    return {
        "n":      len(arr),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "q25":    float(np.percentile(arr, 25)),
        "q75":    float(np.percentile(arr, 75)),
    }


def _section(lines: list, title: str):
    lines.append("")
    lines.append("=" * 65)
    lines.append(f"  {title}")
    lines.append("=" * 65)


def _sig(p: float) -> str:
    if np.isnan(p): return "N/A (datos insuficientes)"
    if p < 0.001: return "*** (p<0.001) → diferencia muy significativa"
    if p < 0.01:  return "**  (p<0.01)"
    if p < 0.05:  return "*   (p<0.05)"
    return "ns  (p≥0.05) → NO hay diferencia significativa"


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def run(df: pl.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("SANITY CHECK 2 — Análisis de posibles valores no útiles")
    lines.append(f"Dataset total: {df.height:,} registros")
    lines.append(f"Columnas disponibles: {', '.join(df.columns)}")

    # Segmentos candidatos a exclusión
    hora0    = df.filter(pl.col("hour") == 0)
    prueba   = df.filter(pl.col("campaign_type") == "prueba")
    resto    = df.filter((pl.col("hour") != 0) & (pl.col("campaign_type") != "prueba"))

    candidatos = df.filter((pl.col("hour") == 0) | (pl.col("campaign_type") == "prueba"))
    n_cand = candidatos.height
    n_resto = resto.height

    lines.append(f"Candidatos a excluir: {n_cand:,} ({n_cand/df.height:.1%})")
    lines.append(f"  → hora 0  : {hora0.height:,}")
    lines.append(f"  → prueba  : {prueba.height:,}")
    lines.append(f"  → solo en resto: {n_resto:,} ({n_resto/df.height:.1%})")

    # -----------------------------------------------------------------------
    # 1. Volumen y distribución temporal
    # -----------------------------------------------------------------------
    _section(lines, "1. DISTRIBUCIÓN HORARIA")

    hour_dist = (
        df.group_by("hour")
        .agg(pl.len().alias("n"))
        .sort("hour")
    )
    lines.append(f"{'Hora':>5}  {'Llamadas':>10}  {'% del total':>12}")
    lines.append("-" * 32)
    for row in hour_dist.iter_rows(named=True):
        marker = " ◄ CANDIDATO" if row["hour"] == 0 else ""
        lines.append(f"{row['hour']:>5}  {row['n']:>10,}  {row['n']/df.height:>11.1%}{marker}")

    # -----------------------------------------------------------------------
    # 2. Tasa de conexión
    # -----------------------------------------------------------------------
    _section(lines, "2. TASA DE CONEXIÓN")

    def conn_rate(sub):
        n = sub.height
        c = sub["connected"].cast(pl.Int32).sum()
        return c, n, c / n if n > 0 else 0

    c_h0, n_h0, r_h0 = conn_rate(hora0)
    c_pr, n_pr, r_pr = conn_rate(prueba)
    c_re, n_re, r_re = conn_rate(resto)

    lines.append(f"  Hora 0   : {r_h0:.1%}  ({c_h0}/{n_h0})")
    lines.append(f"  Prueba   : {r_pr:.1%}  ({c_pr}/{n_pr})")
    lines.append(f"  Resto    : {r_re:.1%}  ({c_re}/{n_re})")

    z_h0, p_h0 = _proportions_ztest(n_h0, c_h0, n_re, c_re)
    z_pr, p_pr = _proportions_ztest(n_pr, c_pr, n_re, c_re)
    p_h0_str = f"{p_h0:.4f}" if not np.isnan(p_h0) else "N/A"
    p_pr_str = f"{p_pr:.4f}" if not np.isnan(p_pr) else "N/A"
    lines.append(f"\n  Z-test hora0 vs resto  : z={z_h0:.2f}  p={p_h0_str}  {_sig(p_h0)}")
    lines.append(f"  Z-test prueba vs resto : z={z_pr:.2f}  p={p_pr_str}  {_sig(p_pr)}")

    # -----------------------------------------------------------------------
    # 3. Duración de llamadas conectadas
    # -----------------------------------------------------------------------
    _section(lines, "3. DURACIÓN DE LLAMADAS CONECTADAS (duration_sec)")

    conn_h0 = hora0.filter(pl.col("connected") == True)
    conn_pr = prueba.filter(pl.col("connected") == True)
    conn_re = resto.filter(pl.col("connected") == True)

    for label, sub in [("Hora 0", conn_h0), ("Prueba", conn_pr), ("Resto", conn_re)]:
        d = _describe(sub["duration_sec"])
        lines.append(
            f"  {label:8s}: n={d['n']:,}  media={d['mean']:.1f}s  "
            f"mediana={d['median']:.1f}s  IQR=[{d['q25']:.0f}, {d['q75']:.0f}]"
        )

    h0_dur = conn_h0["duration_sec"].cast(pl.Float64).drop_nulls().to_list() if conn_h0.height > 0 else []
    re_dur = conn_re["duration_sec"].cast(pl.Float64).drop_nulls().to_list() if conn_re.height > 0 else []
    pr_dur = conn_pr["duration_sec"].cast(pl.Float64).drop_nulls().to_list() if conn_pr.height > 0 else []

    u_h0, p_dur_h0 = _mannwhitney(h0_dur, re_dur)
    u_pr, p_dur_pr = _mannwhitney(pr_dur, re_dur)
    u_h0_str = f"{u_h0:.0f}" if not np.isnan(u_h0) else "N/A"
    p_dur_h0_str = f"{p_dur_h0:.4f}" if not np.isnan(p_dur_h0) else "N/A"
    u_pr_str = f"{u_pr:.0f}" if not np.isnan(u_pr) else "N/A"
    p_dur_pr_str = f"{p_dur_pr:.4f}" if not np.isnan(p_dur_pr) else "N/A"
    lines.append(f"\n  Mann-Whitney hora0 vs resto  : U={u_h0_str}  p={p_dur_h0_str}  {_sig(p_dur_h0)}")
    lines.append(f"  Mann-Whitney prueba vs resto : U={u_pr_str}  p={p_dur_pr_str}  {_sig(p_dur_pr)}")

    # -----------------------------------------------------------------------
    # 4. Distribución de disconnected_reason
    # -----------------------------------------------------------------------
    _section(lines, "4. RAZÓN DE DESCONEXIÓN (llamadas conectadas)")

    for label, sub in [("Hora 0", conn_h0), ("Prueba", conn_pr), ("Resto", conn_re)]:
        dist = (
            sub.group_by("disconnected_reason")
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / sub.height * 100).round(1).alias("pct"))
            .sort("n", descending=True)
        )
        lines.append(f"\n  {label} ({sub.height:,} conectadas):")
        for row in dist.iter_rows(named=True):
            lines.append(f"    {row['disconnected_reason']:30s} {row['n']:5,}  ({row['pct']:.1f}%)")

    # -----------------------------------------------------------------------
    # 5. Distribución de sentimiento
    # -----------------------------------------------------------------------
    _section(lines, "5. SENTIMIENTO PROPIO (llamadas conectadas)")

    for label, sub in [("Hora 0", conn_h0), ("Prueba", conn_pr), ("Resto", conn_re)]:
        dist = (
            sub.group_by("sentiment_own")
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / sub.height * 100).round(1).alias("pct"))
            .sort("n", descending=True)
        )
        lines.append(f"\n  {label} ({sub.height:,} conectadas):")
        for row in dist.iter_rows(named=True):
            lines.append(f"    {row['sentiment_own']:20s} {row['n']:5,}  ({row['pct']:.1f}%)")

    # -----------------------------------------------------------------------
    # 6. Transcripts disponibles
    # -----------------------------------------------------------------------
    _section(lines, "6. CALIDAD DE DATOS (transcripts)")

    def transcript_coverage(sub):
        n = sub.height
        if "transcript_text" not in sub.columns or "transcript_length" not in sub.columns:
            return 0, n, 0.0
        try:
            has_t = sub.filter(pl.col("transcript_text").is_not_null()).filter(
                pl.col("transcript_length") > 0
            ).height
        except Exception:
            has_t = 0
        return has_t, n, has_t / n if n > 0 else 0

    for label, sub in [("Hora 0", hora0), ("Prueba", prueba), ("Resto", resto)]:
        t, n, r = transcript_coverage(sub)
        lines.append(f"  {label:8s}: {r:.1%} con transcript  ({t}/{n})")

    # -----------------------------------------------------------------------
    # 7. Veredicto
    # -----------------------------------------------------------------------
    _section(lines, "7. VEREDICTO")

    evidencia_h0 = []
    if not np.isnan(p_h0) and p_h0 < 0.05:
        diff = r_h0 - r_re
        evidencia_h0.append(f"tasa de conexión diferente ({r_h0:.1%} vs {r_re:.1%}, z={z_h0:.2f})")
    if not np.isnan(p_dur_h0) and p_dur_h0 < 0.05:
        evidencia_h0.append(f"distribución de duración diferente (p={p_dur_h0:.4f})")
    if hora0.height < 50:
        evidencia_h0.append(f"volumen muy bajo ({hora0.height} registros, <50)")

    evidencia_pr = []
    if not np.isnan(p_pr) and p_pr < 0.05:
        evidencia_pr.append(f"tasa de conexión diferente ({r_pr:.1%} vs {r_re:.1%})")
    if not np.isnan(p_dur_pr) and p_dur_pr < 0.05:
        evidencia_pr.append(f"distribución de duración diferente (p={p_dur_pr:.4f})")
    if prueba.height < 50:
        evidencia_pr.append(f"volumen muy bajo ({prueba.height} registros, <50)")

    lines.append("\n  HORA 0 (00:00–00:59):")
    if evidencia_h0:
        lines.append(f"    → RECOMENDACIÓN: EXCLUIR")
        for e in evidencia_h0:
            lines.append(f"       • {e}")
    else:
        lines.append("    → RECOMENDACIÓN: MANTENER (no hay evidencia suficiente para excluir)")

    lines.append("\n  CAMPAÑA 'prueba':")
    if evidencia_pr:
        lines.append(f"    → RECOMENDACIÓN: EXCLUIR")
        for e in evidencia_pr:
            lines.append(f"       • {e}")
    else:
        lines.append("    → RECOMENDACIÓN: MANTENER (no hay evidencia suficiente para excluir)")

    # -----------------------------------------------------------------------
    # Guardar
    # -----------------------------------------------------------------------
    output = "\n".join(lines)
    OUT_FILE.write_text(output, encoding="utf-8")
    print(output)
    print(f"\n[INFO] Resultados guardados en: {OUT_FILE}")


if __name__ == "__main__":
    df = pl.read_csv(CLEAN_CSV)

    # Necesita campaign_type — aplicar clasificación inline si no existe
    if "campaign_type" not in df.columns:
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
        def _classify(name):
            if not name: return "otro"
            n = name.lower()
            for pat, lbl in _CAMPAIGN_PATTERNS:
                if re.search(pat, n): return lbl
            return "otro"
        df = df.with_columns(
            pl.col("name").map_elements(lambda n: _classify(n or ""), return_dtype=pl.Utf8).alias("campaign_type")
        )

    # Necesita sentiment_own
    if "sentiment_own" not in df.columns:
        df = df.with_columns(pl.lit("neutral").alias("sentiment_own"))

    run(df)
