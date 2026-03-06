"""
Sanity Check 4 — Análisis exploratorio del dataset limpiado.

Carga calls_clean.csv y produce:
  1. Tabla de tipos de dato y valores nulos (todas las columnas)
  2. Distribución de columnas de baja cardinalidad (posibles categorías)
  3. Estadísticas descriptivas de columnas numéricas
  4. Correlación Spearman entre variables numéricas
  5. η² (eta squared) entre numéricas y categóricas
  6. Cramér's V entre categóricas/binarias
  7. Recomendaciones para clustering (features relevantes)

Salida texto : data/interim/sc4_clean_analysis.txt
Salida graficos: data/interim/sc4_*.png

Detección automática de columnas:
  - IDs: excluidas (terminan en _id o son identificadores conocidos)
  - Textos largos / técnicas: excluidas explícitamente
  - Numéricas: dtype Int*/Float* con cardinalidad > CAT_THRESHOLD
  - Categóricas: dtype Utf8/Boolean/Categorical con cardinalidad <= CAT_THRESHOLD
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rich import box
from rich.console import Console
from rich.table import Table
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "calls_clean.csv"
OUT_DIR   = PROJECT_ROOT / "data" / "interim"
OUT_FILE  = OUT_DIR / "sc4_clean_analysis.txt"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Umbral de cardinalidad para separar categórica vs numérica (en columnas ambiguas)
CAT_THRESHOLD = 33

# Columnas a omitir completamente (textos largos, URLs, timestamps, técnicas)
SKIP_COLS = {
    "name", "call_url",
    "transcript", "post_call_analysis", "transcript_text",
    "date", "executed_at", "pca_resumen",
    "duration_ms",  # redundante con duration_sec
}

# Columnas numéricas con alta cardinalidad que son en realidad identificadores
NUMERIC_SKIP_COLS = {"day_of_month"}  # día del mes tiene baja relevancia para clustering

# Umbrales para recomendación de clustering
ETA2_THRESHOLD = 0.06   # η² ≥ 0.06 → asociación mediana o mayor
V_THRESHOLD    = 0.20   # Cramér's V ≥ 0.20 → asociación débil-moderada o mayor
SPEAR_THRESHOLD = 0.30  # |r| ≥ 0.30 → correlación notable


def _is_id_col(col: str) -> bool:
    """Retorna True si la columna parece un identificador."""
    return col.endswith("_id") or col in {"id", "call_id", "contact_id", "target_id", "campaign_id"}


def _classify_columns(df: pl.DataFrame):
    """
    Separa automáticamente las columnas del DataFrame en:
      - num_cols : numéricas continuas/discretas de interés
      - cat_cols : categóricas/binarias de baja cardinalidad (sin binarias bool puras)
      - bool_cols: columnas boolean o binarias (0/1)
    Excluye IDs, textos largos y columnas en SKIP_COLS.
    """
    num_cols  = []
    cat_cols  = []
    bool_cols = []

    for col in df.columns:
        # Excluir IDs y columnas técnicas
        if _is_id_col(col) or col in SKIP_COLS:
            continue

        dtype     = df[col].dtype
        n_unique  = df[col].n_unique()

        # Boolean puro → binaria
        if dtype == pl.Boolean:
            bool_cols.append(col)
            continue

        # Entero/float con solo 2 valores distintos → tratar como binaria
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64) and n_unique <= 2:
            bool_cols.append(col)
            continue

        # Numérico con cardinalidad alta → numérica continua
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64):
            if col in NUMERIC_SKIP_COLS:
                continue
            num_cols.append(col)
            continue

        # Utf8 / Categorical / Enum con cardinalidad baja → categórica
        if dtype in (pl.Utf8, pl.String, pl.Categorical) or str(dtype).startswith("Enum"):
            if n_unique <= CAT_THRESHOLD:
                cat_cols.append(col)
            # cardinalidad alta en texto → omitir (texto libre)
            continue

    return num_cols, cat_cols, bool_cols


# ---------------------------------------------------------------------------
# Helpers estadísticos
# ---------------------------------------------------------------------------

def _section(lines: list, title: str) -> None:
    lines.append("")
    lines.append("=" * 75)
    lines.append(f"  {title}")
    lines.append("=" * 75)


def cramers_v(s1: pl.Series, s2: pl.Series) -> float:
    """Cramér's V entre dos series categóricas (maneja nulos)."""
    df_pair = pl.DataFrame({"a": s1.cast(pl.Utf8), "b": s2.cast(pl.Utf8)}).drop_nulls()
    if df_pair.height < 10:
        return float("nan")

    cats_a = sorted(df_pair["a"].unique().to_list())
    cats_b = sorted(df_pair["b"].unique().to_list())
    idx_a  = {v: i for i, v in enumerate(cats_a)}
    idx_b  = {v: i for i, v in enumerate(cats_b)}

    table = np.zeros((len(cats_a), len(cats_b)), dtype=int)
    for a, b in zip(df_pair["a"].to_list(), df_pair["b"].to_list()):
        table[idx_a[a], idx_b[b]] += 1

    try:
        chi2, _, _, _ = stats.chi2_contingency(table)
        n = table.sum()
        k = min(table.shape) - 1
        if k == 0 or n == 0:
            return float("nan")
        return float(min(np.sqrt(chi2 / (n * k)), 1.0))
    except Exception:
        return float("nan")


def eta_squared(num: pl.Series, cat: pl.Series) -> float:
    """η² (eta squared) entre una variable numérica y una categórica."""
    df_pair = pl.DataFrame({"n": num, "c": cat.cast(pl.Utf8)}).drop_nulls()
    if df_pair.height < 10:
        return float("nan")

    grand_mean = df_pair["n"].cast(pl.Float64).mean()
    ss_total   = ((df_pair["n"].cast(pl.Float64) - grand_mean) ** 2).sum()
    if not ss_total or ss_total == 0:
        return float("nan")

    ss_between = sum(
        grp.height * (grp["n"].cast(pl.Float64).mean() - grand_mean) ** 2
        for grp in [
            df_pair.filter(pl.col("c") == v)
            for v in df_pair["c"].unique().to_list()
        ]
    )
    return float(min(ss_between / ss_total, 1.0))


def _fmt(v: float) -> str:
    return f"{v:.3f}" if not np.isnan(v) else "  nan"


# ---------------------------------------------------------------------------
# Análisis principal
# ---------------------------------------------------------------------------

def run(df: pl.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("SANITY CHECK 4 — Análisis exploratorio del dataset limpiado")
    lines.append(f"Registros: {df.height:,} | Columnas: {df.width}")

    # Clasificación automática de columnas
    num_present, cat_present, bool_present = _classify_columns(df)
    all_cats = cat_present + bool_present

    skipped_ids  = [c for c in df.columns if _is_id_col(c)]
    skipped_text = [c for c in df.columns if c in SKIP_COLS]

    console.print(f"\n[bold]Columnas clasificadas automáticamente:[/bold]")
    console.print(f"  [green]Numéricas ({len(num_present)}):[/green]  {num_present}")
    console.print(f"  [yellow]Categóricas ({len(cat_present)}):[/yellow] {cat_present}")
    console.print(f"  [cyan]Binarias ({len(bool_present)}):[/cyan]   {bool_present}")
    console.print(f"  [dim]IDs omitidas ({len(skipped_ids)}):[/dim] {skipped_ids}")
    console.print(f"  [dim]Texto/técnicas omitidas ({len(skipped_text)}):[/dim] {skipped_text}\n")

    # -----------------------------------------------------------------------
    # 1. Tipos de dato y valores nulos
    # -----------------------------------------------------------------------
    _section(lines, "1. TIPOS DE DATO Y VALORES NULOS")

    tbl = Table(
        title="Tipos de Dato y Valores Nulos",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    tbl.add_column("Columna",           style="bold",     min_width=26)
    tbl.add_column("Tipo",              justify="center", min_width=10)
    tbl.add_column("Nulos",             justify="right",  min_width=7)
    tbl.add_column("% Nulos",           justify="right",  min_width=8)
    tbl.add_column("Válidos",           justify="right",  min_width=8)
    tbl.add_column("% Válidos",         justify="right",  min_width=9)
    tbl.add_column("Valores Distintos", justify="right",  min_width=18)

    lines.append(
        f"{'Columna':<26} {'Tipo':>11} {'Nulos':>8} {'%Nulos':>8} "
        f"{'Válidos':>8} {'%Válidos':>9} {'Distintos':>10}"
    )
    lines.append("-" * 92)

    for col in df.columns:
        dtype      = str(df[col].dtype)
        nulls      = df[col].null_count()
        valid      = df.height - nulls
        pct_null   = nulls / df.height * 100
        pct_valid  = valid / df.height * 100
        n_distinct = df[col].n_unique()

        row_style = "red" if pct_null > 50 else "yellow" if pct_null > 10 else ""
        tbl.add_row(
            col, dtype[:11],
            f"{nulls:,}", f"{pct_null:.1f}%",
            f"{valid:,}", f"{pct_valid:.1f}%",
            f"{n_distinct:,}",
            style=row_style,
        )
        lines.append(
            f"{col:<26} {dtype[:11]:>11} {nulls:>8,} {pct_null:>7.1f}% "
            f"{valid:>8,} {pct_valid:>8.1f}% {n_distinct:>10,}"
        )

    console.print(tbl)

    # -----------------------------------------------------------------------
    # 2. Columnas de baja cardinalidad (categóricas + binarias detectadas)
    # -----------------------------------------------------------------------
    _section(lines, f"2. COLUMNAS DE BAJA CARDINALIDAD (≤ {CAT_THRESHOLD} valores distintos)")
    lines.append("  Distribución de valores para columnas detectadas como categóricas o binarias\n")

    low_card = all_cats  # usa la clasificación automática

    for col in low_card:
        dtype      = str(df[col].dtype)
        n_distinct = df[col].n_unique()
        vc = (
            df.group_by(col)
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / df.height * 100).round(1).alias("pct"))
            .sort("n", descending=True)
        )

        header = f"  [{col}]  tipo={dtype}  distintos={n_distinct}"
        lines.append(header)
        console.print(f"\n[bold cyan]{header}[/bold cyan]")

        cat_tbl = Table(box=box.SIMPLE, show_header=True, header_style="dim")
        cat_tbl.add_column("Valor",    min_width=22)
        cat_tbl.add_column("Cantidad", justify="right", min_width=10)
        cat_tbl.add_column("%",        justify="right", min_width=7)

        for row in vc.iter_rows(named=True):
            val = str(row[col]) if row[col] is not None else "(null)"
            cat_tbl.add_row(val, f"{row['n']:,}", f"{row['pct']:.1f}%")
            lines.append(f"    {val:<25} {row['n']:>8,}  {row['pct']:>6.1f}%")

        console.print(cat_tbl)
        lines.append("")

    # -----------------------------------------------------------------------
    # 3. Estadísticas descriptivas de numéricas
    # -----------------------------------------------------------------------
    _section(lines, "3. ESTADÍSTICAS DESCRIPTIVAS — VARIABLES NUMÉRICAS")

    num_tbl = Table(box=box.ROUNDED, header_style="bold cyan")
    for col_h in ["Columna", "Media", "Mediana", "Std", "Min", "Max", "P25", "P75"]:
        num_tbl.add_column(col_h, justify="right" if col_h != "Columna" else "left", min_width=10)

    lines.append(
        f"  {'Columna':<20} {'Media':>10} {'Mediana':>10} {'Std':>10} "
        f"{'Min':>8} {'Max':>8} {'P25':>8} {'P75':>8}"
    )
    lines.append("  " + "-" * 84)

    for col in num_present:
        arr = df[col].drop_nulls().cast(pl.Float64).to_numpy()
        if len(arr) == 0:
            continue
        mean, median, std = float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))
        mn, mx            = float(np.min(arr)), float(np.max(arr))
        p25, p75          = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

        num_tbl.add_row(
            col, f"{mean:.1f}", f"{median:.1f}", f"{std:.1f}",
            f"{mn:.0f}", f"{mx:.0f}", f"{p25:.1f}", f"{p75:.1f}",
        )
        lines.append(
            f"  {col:<20} {mean:>10.1f} {median:>10.1f} {std:>10.1f} "
            f"{mn:>8.0f} {mx:>8.0f} {p25:>8.1f} {p75:>8.1f}"
        )

    console.print(num_tbl)

    # -----------------------------------------------------------------------
    # 4. Correlación Spearman entre numéricas
    # -----------------------------------------------------------------------
    _section(lines, "4. CORRELACIÓN SPEARMAN — VARIABLES NUMÉRICAS")

    n_num        = len(num_present)
    corr_matrix  = np.full((n_num, n_num), np.nan)

    for i, c1 in enumerate(num_present):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n_num):
            c2      = num_present[j]
            df_pair = df.select([c1, c2]).drop_nulls()
            if df_pair.height < 10:
                continue
            r, _ = stats.spearmanr(
                df_pair[c1].cast(pl.Float64).to_numpy(),
                df_pair[c2].cast(pl.Float64).to_numpy(),
            )
            corr_matrix[i, j] = corr_matrix[j, i] = float(r)

    sp_tbl = Table(box=box.SIMPLE, header_style="bold")
    sp_tbl.add_column("", min_width=18)
    for col in num_present:
        sp_tbl.add_column(col[:14], justify="right", min_width=14)

    lines.append(f"  {'':18}" + "".join(f"{c[:14]:>15}" for c in num_present))
    for i, c1 in enumerate(num_present):
        row_vals = [_fmt(corr_matrix[i, j]) for j in range(n_num)]
        sp_tbl.add_row(c1, *row_vals)
        lines.append(f"  {c1:<18}" + "".join(f"{v:>15}" for v in row_vals))

    console.print(sp_tbl)

    # -----------------------------------------------------------------------
    # 5. η² numérica ↔ categórica
    # -----------------------------------------------------------------------
    _section(lines, "5. ASOCIACIÓN NUMÉRICA ↔ CATEGÓRICA (η² eta squared)")
    lines.append(
        "  Interpretación: η²≈0 sin efecto | ≈0.01 pequeño | ≈0.06 mediano | ≥0.14 grande\n"
    )

    eta_tbl = Table(box=box.ROUNDED, header_style="bold cyan")
    eta_tbl.add_column("Numérica \\ Categórica", min_width=20)
    for cat in cat_present:
        eta_tbl.add_column(cat[:16], justify="right", min_width=16)

    label = "Numérica \\ Categórica"
    lines.append(f"  {label:<20}" + "".join(f"{c[:16]:>18}" for c in cat_present))
    lines.append("  " + "-" * (20 + 18 * len(cat_present)))

    for num_col in num_present:
        row_vals = [_fmt(eta_squared(df[num_col], df[cat_col])) for cat_col in cat_present]
        eta_tbl.add_row(num_col, *row_vals)
        lines.append(f"  {num_col:<20}" + "".join(f"{v:>18}" for v in row_vals))

    console.print(eta_tbl)

    # -----------------------------------------------------------------------
    # 6. Cramér's V entre categóricas y binarias
    # -----------------------------------------------------------------------
    _section(lines, "6. ASOCIACIÓN ENTRE CATEGÓRICAS/BINARIAS (Cramér's V)")
    lines.append(
        "  Interpretación: V≈0 sin asociación | 0.1–0.3 débil | 0.3–0.7 moderada | >0.7 fuerte\n"
    )

    n_cats   = len(all_cats)
    v_matrix = np.full((n_cats, n_cats), np.nan)

    for i in range(n_cats):
        v_matrix[i, i] = 1.0
        for j in range(i + 1, n_cats):
            v = cramers_v(df[all_cats[i]], df[all_cats[j]])
            v_matrix[i, j] = v_matrix[j, i] = v

    cv_tbl = Table(box=box.SIMPLE, header_style="bold")
    cv_tbl.add_column("", min_width=24)
    for col in all_cats:
        cv_tbl.add_column(col[:13], justify="right", min_width=13)

    lines.append(f"  {'':24}" + "".join(f"{c[:13]:>15}" for c in all_cats))
    for i, c1 in enumerate(all_cats):
        row_vals = [_fmt(v_matrix[i, j]) for j in range(n_cats)]
        cv_tbl.add_row(c1, *row_vals)
        lines.append(f"  {c1:<24}" + "".join(f"{v:>15}" for v in row_vals))

    console.print(cv_tbl)

    # -----------------------------------------------------------------------
    # 7. Recomendaciones para clustering
    # -----------------------------------------------------------------------
    _section(lines, "7. RECOMENDACIONES PARA CLUSTERING")
    lines.append(
        "  Features sugeridas según correlaciones/asociaciones detectadas.\n"
        "  Para datos mixtos (numéricas + categóricas) se recomienda K-Prototypes.\n"
        "  Para solo numéricas se recomienda K-Means o DBSCAN (previo StandardScaler).\n"
    )

    # a) Numéricas relevantes: baja correlación entre sí (no redundantes), alta asociación con categóricas
    lines.append("  [A] NUMÉRICAS CANDIDATAS A FEATURES")
    num_candidates = []
    for i, c1 in enumerate(num_present):
        # Verificar que no sea redundante con otra numérica ya incluida (|r| < 0.85)
        redundant = False
        for j, c2 in enumerate(num_candidates):
            idx_c2 = num_present.index(c2)
            r = corr_matrix[i, idx_c2] if not np.isnan(corr_matrix[i, idx_c2]) else 0
            if abs(r) >= 0.85:
                redundant = True
                lines.append(f"    - {c1}: REDUNDANTE con {c2} (r={r:.2f}) → omitir")
                break
        if not redundant:
            # Verificar si tiene asociación con alguna categórica (η² ≥ umbral)
            eta_vals = [eta_squared(df[c1], df[cat]) for cat in cat_present]
            max_eta  = max((v for v in eta_vals if not np.isnan(v)), default=0.0)
            tag = f"η²_max={max_eta:.3f}"
            num_candidates.append(c1)
            mark = "✓" if max_eta >= ETA2_THRESHOLD else "~"
            lines.append(f"    {mark} {c1}: {tag}")

    lines.append("")
    lines.append("  [B] CATEGÓRICAS CANDIDATAS A FEATURES (Cramér's V con otras categorías)")
    cat_candidates = []
    for col in cat_present:
        idx = all_cats.index(col)
        # V máxima con otras categóricas (excluyendo diagonal)
        v_vals = [
            v_matrix[idx, j]
            for j in range(n_cats)
            if j != idx and not np.isnan(v_matrix[idx, j])
        ]
        max_v = max(v_vals, default=0.0)
        n_u   = df[col].n_unique()
        mark  = "✓" if max_v >= V_THRESHOLD else "~"
        cat_candidates.append(col)
        lines.append(f"    {mark} {col}: V_max={max_v:.3f}, distintos={n_u}")

    lines.append("")
    lines.append("  [C] BINARIAS CANDIDATAS (Cramér's V con categóricas)")
    for col in bool_present:
        idx   = all_cats.index(col)
        v_vals = [
            v_matrix[idx, j]
            for j in range(n_cats)
            if j != idx and not np.isnan(v_matrix[idx, j])
        ]
        max_v = max(v_vals, default=0.0)
        mark  = "✓" if max_v >= V_THRESHOLD else "~"
        lines.append(f"    {mark} {col}: V_max={max_v:.3f}")

    lines.append("")
    lines.append("  [D] RESUMEN — FEATURES RECOMENDADAS PARA CLUSTERING")
    strong_num = [c for c in num_candidates if any(
        not np.isnan(eta_squared(df[c], df[cat])) and eta_squared(df[c], df[cat]) >= ETA2_THRESHOLD
        for cat in cat_present
    )]
    def _max_v_for(col: str) -> float:
        idx = all_cats.index(col)
        vals = [v_matrix[idx, j] for j in range(n_cats) if j != idx and not np.isnan(v_matrix[idx, j])]
        return max(vals, default=0.0)

    strong_cat = [c for c in cat_present if _max_v_for(c) >= V_THRESHOLD]
    recommended = strong_num + strong_cat
    lines.append(f"    Numéricas  : {strong_num}")
    lines.append(f"    Categóricas: {strong_cat}")
    lines.append(f"    Total      : {len(recommended)} features")
    lines.append(
        f"    Algoritmo recomendado: {'K-Prototypes (mixto)' if strong_cat else 'K-Means / DBSCAN (solo numéricas)'}"
    )

    _rec_tbl = Table(box=box.ROUNDED, header_style="bold cyan", title="Features recomendadas para clustering")
    _rec_tbl.add_column("Feature",   min_width=28)
    _rec_tbl.add_column("Tipo",      min_width=12)
    _rec_tbl.add_column("Motivo",    min_width=40)
    for c in strong_num:
        eta_vals = [eta_squared(df[c], df[cat]) for cat in cat_present]
        max_eta  = max((v for v in eta_vals if not np.isnan(v)), default=0.0)
        _rec_tbl.add_row(c, "numérica", f"η²_max={max_eta:.3f} ≥ {ETA2_THRESHOLD}", style="green")
    for c in strong_cat:
        idx   = all_cats.index(c)
        v_vals = [v_matrix[idx, j] for j in range(n_cats) if j != idx and not np.isnan(v_matrix[idx, j])]
        max_v  = max(v_vals, default=0.0)
        _rec_tbl.add_row(c, "categórica", f"V_max={max_v:.3f} ≥ {V_THRESHOLD}", style="yellow")
    console.print(_rec_tbl)

    # -----------------------------------------------------------------------
    # Guardar reporte
    # -----------------------------------------------------------------------
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[INFO] Reporte guardado en: {OUT_FILE}")

    # -----------------------------------------------------------------------
    # Gráficos
    # -----------------------------------------------------------------------
    _create_plots(df, num_present, corr_matrix, cat_present, all_cats, v_matrix, low_card)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _create_plots(
    df: pl.DataFrame,
    num_cols: list,
    corr_matrix: np.ndarray,
    cat_cols: list,
    all_cats: list,
    v_matrix: np.ndarray,
    low_card: list,
) -> None:

    # 1. Heatmap Spearman (numéricas)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Spearman r")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(num_cols, fontsize=10)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            v = corr_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Correlación Spearman — Variables Numéricas", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "sc4_spearman_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[INFO] Gráfico: sc4_spearman_heatmap.png")

    # 2. Heatmap Cramér's V (categóricas + binarias)
    fig, ax = plt.subplots(figsize=(max(8, len(all_cats) * 1.1), max(7, len(all_cats) * 0.9)))
    im = ax.imshow(v_matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Cramér's V")
    ax.set_xticks(range(len(all_cats)))
    ax.set_yticks(range(len(all_cats)))
    ax.set_xticklabels(all_cats, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(all_cats, fontsize=9)
    for i in range(len(all_cats)):
        for j in range(len(all_cats)):
            v = v_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.6 else "black")
    ax.set_title("Cramér's V — Variables Categóricas y Binarias", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "sc4_cramersv_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[INFO] Gráfico: sc4_cramersv_heatmap.png")

    # 3. Barras horizontales para columnas de baja cardinalidad
    if not low_card:
        return

    ncols  = 2
    nrows  = (len(low_card) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes_flat = np.array(axes).flatten()

    for idx, col in enumerate(low_card):
        ax = axes_flat[idx]
        vc = (
            df.group_by(col)
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        vals   = [str(v) if v is not None else "(null)" for v in vc[col].to_list()]
        counts = vc["n"].to_list()
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(vals)))

        ax.barh(vals[::-1], counts[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
        ax.set_title(f"{col}  ({len(vals)} distintos)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Cantidad")
        ax.grid(axis="x", alpha=0.3)

        for bar, count in zip(ax.patches, counts[::-1]):
            pct = count / df.height * 100
            ax.text(
                bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {count:,} ({pct:.1f}%)", va="center", fontsize=8,
            )

    # Ocultar ejes sobrantes
    for idx in range(len(low_card), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(
        "Distribución de Variables de Baja Cardinalidad",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "sc4_low_cardinality.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[INFO] Gráfico: sc4_low_cardinality.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not CLEAN_CSV.exists():
        print(f"[ERROR] No se encontró el dataset limpiado en: {CLEAN_CSV}")
        print("        Ejecuta el pipeline primero: python scripts/clean_data.py")
        sys.exit(1)

    print(f"[INFO] Cargando {CLEAN_CSV.name}...")
    df = pl.read_csv(CLEAN_CSV, infer_schema_length=5000)
    print(f"[INFO] Dataset cargado: {df.height:,} registros, {df.width} columnas\n")

    run(df)
