"""
Script 05b — Clustering de contactos conectados (v2: K-Means + encoding de dominio).

Diferencias clave vs cluster_contacts.py (v1 / K-Prototypes):
  - Algoritmo  : K-Means sklearn (C-optimizado) vs kmodes (Python puro) → ~10-100x más rápido
  - Encoding   : ordinal de dominio para categóricas → todo float64 (sin strings en la matriz)
  - K_RANGE    : 2-5 (v1 usaba 2-7)
  - Silhouette : sobre la matriz completa (v1 solo usaba parte numérica como aproximación)
  - Sin baseline comparativo K-Means (ya es K-Means)

Features utilizadas (6 total):
  Numéricas   : duration_sec, hour
  Categóricas → numéricas con encoding de dominio:
    disconnected_reason → engagement {user_hangup:2, agent_hangup:1, resto:0}
    pca_sentimiento     → ordinal    {positivo:2, neutral:1, negativo:0, null→1}
    pca_razon_churn     → binaria    tiene_motivo (0/1)
  Binaria     : inconsistency_flag
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from utils.paths import CLEAN_CSV, PROCESSED_DIR, FIGURES_DIR
from utils.plotting import savefig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
K_RANGE      = range(2, 6)   # 4 valores; suficiente para ~13K conectados
SAMPLE_FRAC  = 0.15           # muestra para selección de K

# Encoding de dominio
ENGAGEMENT_MAP = {"user_hangup": 2, "agent_hangup": 1}          # resto → 0
SENTIMENT_MAP  = {"positivo": 2, "neutral": 1, "negativo": 0}   # null → 1 (neutral)
SIN_MOTIVO     = {None, "", "n/a", "otro"}                       # valores sin motivo churn

# Columnas clave: se excluyen del entrenamiento filas con nulos en estas.
# pca_sentimiento incluida — su null no se imputa como neutral en el modelo.
# pca_razon_churn NO incluida (~99.8% null) — su null = "sin_motivo" es categoría válida.
CLUSTER_KEY_COLS = ["duration_sec", "hour", "disconnected_reason", "pca_sentimiento"]

# Salidas (sufijo _v2 para no pisar v1)
FIG_K_SELECTION = "cluster_v2_k_selection.png"
FIG_PROFILES    = "cluster_v2_profiles.png"
FIG_RAZON_DIST  = "cluster_v2_razon_dist.png"
CLUSTERS_CSV    = "clusters_contacts_v2.csv"

# ---------------------------------------------------------------------------
# Encoding de features
# ---------------------------------------------------------------------------

def _encode_features(df: pl.DataFrame, dur_fill: float = 0.0) -> np.ndarray:
    """
    Convierte las 6 features a float64 con encoding de dominio.
    Retorna ndarray (n, 6) sin escalar.

    dur_fill: valor de imputación para duration_sec (pasar mediana del conjunto
              de entrenamiento; irrelevante cuando df ya viene sin nulos).
    """
    # 1. duration_sec — imputación solo activa para el pase de predicción
    duration = df["duration_sec"].fill_null(dur_fill).to_numpy().astype(float)

    # 2. hour
    hour = df["hour"].to_numpy().astype(float)

    # 3. disconnected_reason → jerarquía de engagement
    engagement = np.array(
        [ENGAGEMENT_MAP.get(str(v) if v is not None else "", 0)
         for v in df["disconnected_reason"].to_list()],
        dtype=float,
    )

    # 4. pca_sentimiento → ordinal (null → 1 = neutral)
    sentiment = np.array(
        [SENTIMENT_MAP.get(str(v) if v is not None else "", 1)
         for v in df["pca_sentimiento"].to_list()],
        dtype=float,
    )

    # 5. pca_razon_churn → binaria tiene_motivo
    tiene_motivo = np.array(
        [0.0 if v in SIN_MOTIVO else 1.0
         for v in df["pca_razon_churn"].to_list()],
        dtype=float,
    )

    # 6. inconsistency_flag → 0/1
    inconsistency = df["inconsistency_flag"].cast(pl.Float64).to_numpy()

    return np.column_stack([duration, hour, engagement, sentiment, tiene_motivo, inconsistency])


# ---------------------------------------------------------------------------
# Selección de K (elbow + silhouette sobre muestra)
# ---------------------------------------------------------------------------

def _select_k(X_scaled: np.ndarray) -> tuple[int, dict, dict]:
    rng      = np.random.default_rng(RANDOM_STATE)
    n_sample = max(int(X_scaled.shape[0] * SAMPLE_FRAC), 500)
    n_sample = min(n_sample, X_scaled.shape[0])
    idx      = rng.choice(X_scaled.shape[0], size=n_sample, replace=False)
    X_sample = X_scaled[idx]
    print(f"[INFO] Muestra selección K: {n_sample:,} registros ({SAMPLE_FRAC:.0%})")

    inertias   = {}
    sil_scores = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task("[yellow]Evaluando K...[/]", total=len(K_RANGE))

        for k in K_RANGE:
            km     = KMeans(n_clusters=k, init="k-means++", n_init=10,
                            random_state=RANDOM_STATE)
            labels = km.fit_predict(X_sample)

            inertias[k]   = float(km.inertia_)
            sil           = silhouette_score(X_sample, labels) if len(set(labels)) > 1 else -1.0
            sil_scores[k] = float(sil)

            prog.update(task, advance=1,
                        description=f"[yellow]K={k} | inertia={km.inertia_:.0f} | sil={sil:.3f}[/]")

    best_k = max(sil_scores, key=sil_scores.get)
    return best_k, inertias, sil_scores


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _plot_k_selection(inertias: dict, sil_scores: dict, best_k: int, out_path: Path) -> None:
    ks  = sorted(inertias.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, [inertias[k] for k in ks], "o-", color="#1565C0", linewidth=2)
    ax1.axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    ax1.set_title("Elbow — Inercia K-Means", fontweight="bold")
    ax1.set_xlabel("Número de clusters k")
    ax1.set_ylabel("Inercia (SSE)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ks, [sil_scores[k] for k in ks], "s-", color="#2E7D32", linewidth=2)
    ax2.axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("Número de clusters k")
    ax2.set_ylabel("Silhouette")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Selección del número óptimo de clusters (v2 — K-Means)", fontsize=13, fontweight="bold")
    savefig(fig, out_path, report=True)


def _pct_col(df_c: pl.DataFrame, col: str, val, cluster: int) -> float:
    """% de registros con df_c[col]==val dentro de un cluster."""
    sub   = df_c.filter(pl.col("cluster_id") == cluster)
    total = sub.height or 1
    return sub.filter(pl.col(col) == val).height / total * 100


def _plot_profiles(df_c: pl.DataFrame, out_path: Path) -> None:
    """
    Fila 0: boxplots de duration_sec y hour + tabla de tamaños
    Fila 1: stacked bars de disconnected_reason, pca_sentimiento, tiene_motivo_str
    """
    clusters = sorted(df_c["cluster_id"].unique().to_list())
    n_cl     = len(clusters)
    colors   = plt.cm.Set2(np.linspace(0, 0.8, n_cl))

    # Columna auxiliar legible para motivo churn
    df_c = df_c.with_columns(
        pl.when(
            pl.col("pca_razon_churn").is_null() |
            pl.col("pca_razon_churn").is_in(["n/a", "otro", ""])
        )
        .then(pl.lit("sin_motivo"))
        .otherwise(pl.lit("con_motivo"))
        .alias("tiene_motivo_str")
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # ---- Fila 0: boxplots numéricos ----
    for ax_i, col in enumerate(["duration_sec", "hour"]):
        ax   = axes[0, ax_i]
        data = [df_c.filter(pl.col("cluster_id") == cl)[col].drop_nulls().to_list()
                for cl in clusters]
        bp = ax.boxplot(data, labels=[f"C{c}" for c in clusters],
                        patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)

    # Tabla de tamaños
    ax_sz = axes[0, 2]
    ax_sz.axis("off")
    sizes = df_c.group_by("cluster_id").agg(pl.len().alias("n")).sort("cluster_id")
    tdata = [[f"C{r['cluster_id']}", f"{r['n']:,}", f"{r['n']/df_c.height*100:.1f}%"]
             for r in sizes.iter_rows(named=True)]
    tbl   = ax_sz.table(cellText=tdata, colLabels=["Cluster", "N", "%"],
                         loc="center", cellLoc="center")
    tbl.scale(1.2, 1.8)
    ax_sz.set_title("Tamaño de clusters", fontsize=10, fontweight="bold")
    axes[0, 3].set_visible(False)

    # ---- Fila 1: stacked bars categóricas ----
    cat_specs = [
        ("disconnected_reason", "Razón desconexión"),
        ("pca_sentimiento",     "Sentimiento (pca)"),
        ("tiene_motivo_str",    "Motivo churn"),
    ]

    for ax_i, (col, title) in enumerate(cat_specs):
        ax   = axes[1, ax_i]
        vals = sorted(df_c[col].drop_nulls().unique().to_list())
        cmap = plt.cm.Set3(np.linspace(0, 1, max(len(vals), 1)))

        bottom = np.zeros(n_cl)
        for vi, val in enumerate(vals):
            heights = [_pct_col(df_c, col, val, cl) for cl in clusters]
            ax.bar(clusters, heights, bottom=bottom, label=str(val),
                   color=cmap[vi], edgecolor="white", linewidth=0.5)
            bottom += np.array(heights)

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("%" if ax_i == 0 else "")
        ax.set_xticks(clusters)
        ax.legend(fontsize=7, loc="upper right")

    axes[1, 3].set_visible(False)

    plt.suptitle("Perfiles de Clusters — Variables de Negocio (v2)", fontsize=13, fontweight="bold")
    savefig(fig, out_path, report=True)


def _plot_razon_dist(df_c: pl.DataFrame, out_path: Path) -> None:
    """Heatmap % de razón churn real (top valores, sin null/otro/n-a) por cluster."""
    clusters = sorted(df_c["cluster_id"].unique().to_list())
    n_cl     = len(clusters)

    razones = (
        df_c.filter(pl.col("pca_razon_churn").is_not_null())
            .filter(~pl.col("pca_razon_churn").is_in(["n/a", "otro", ""]))
            ["pca_razon_churn"].value_counts()
            .sort("count", descending=True)
            .head(10)["pca_razon_churn"].to_list()
    )
    if not razones:
        print("[WARN] Sin datos suficientes para cluster_v2_razon_dist.png")
        return

    matrix = np.array([
        [_pct_col(df_c, "pca_razon_churn", r, cl) for cl in clusters]
        for r in razones
    ])

    fig, ax = plt.subplots(figsize=(max(6, n_cl * 2), max(5, len(razones) * 0.7)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="%")
    ax.set_xticks(range(n_cl))
    ax.set_yticks(range(len(razones)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_yticklabels(razones)

    for gi in range(len(razones)):
        for ci in range(n_cl):
            v = matrix[gi, ci]
            ax.text(ci, gi, f"{v:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if v > 50 else "black")

    ax.set_title("Distribución de razón churn por cluster (%) — v2", fontweight="bold")
    savefig(fig, out_path, report=True)


# ---------------------------------------------------------------------------
# Entrenamiento final y exportación
# ---------------------------------------------------------------------------

def _train_kmeans(X_scaled: np.ndarray, best_k: int) -> KMeans:
    print(f"[INFO] Entrenando K-Means final (k={best_k}, subconjunto limpio)...")
    km = KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE)
    km.fit(X_scaled)
    return km


def _export_clusters(df_c: pl.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / CLUSTERS_CSV
    df_c.write_csv(out)
    print(f"[INFO] Exportado: {out.name}  ({df_c.height:,} registros, {df_c['cluster_id'].n_unique()} clusters)")


# ---------------------------------------------------------------------------
# Función principal del pipeline
# ---------------------------------------------------------------------------

def cluster_contacts(df: pl.DataFrame) -> pl.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Solo conectados
    df_conn = df.filter(pl.col("connected") == True)
    print(f"[INFO] Registros conectados: {df_conn.height:,}")

    # Subconjunto limpio para entrenamiento (sin nulos en columnas clave)
    null_mask = pl.lit(True)
    for col in CLUSTER_KEY_COLS:
        null_mask = null_mask & pl.col(col).is_not_null()
    df_train = df_conn.filter(null_mask)
    n_dropped = df_conn.height - df_train.height
    print(f"[INFO] Registros para entrenamiento (sin nulos en cols clave): {df_train.height:,}  (excluidos: {n_dropped:,})")

    # Encoding → escalado: fit sobre subconjunto limpio, transform sobre todos.
    # dur_fill se computa desde df_train para no contaminar la escala con nulls.
    dur_fill    = float(df_train["duration_sec"].median() or 0.0)
    X_raw_train = _encode_features(df_train)                        # sin nulls → dur_fill irrelevante
    X_raw_all   = _encode_features(df_conn, dur_fill=dur_fill)      # imputa nulls solo en pase de predicción
    scaler      = StandardScaler().fit(X_raw_train)
    X_train     = scaler.transform(X_raw_train)
    X_all       = scaler.transform(X_raw_all)

    # Selección de K (sobre muestra del subconjunto limpio)
    print("[INFO] Evaluando K (elbow + silhouette sobre muestra)...")
    best_k, inertias, sil_scores = _select_k(X_train)
    print(f"[INFO] K sugerido: {best_k}  (sil={sil_scores[best_k]:.4f})")
    _plot_k_selection(inertias, sil_scores, best_k, FIGURES_DIR / FIG_K_SELECTION)

    # Modelo final entrenado sobre subconjunto limpio, predicción sobre todos
    km           = _train_kmeans(X_train, best_k)
    final_labels = km.predict(X_all)
    sil_full     = silhouette_score(X_all, final_labels) if len(set(final_labels)) > 1 else -1.0
    print(f"[INFO] Silhouette final (dataset completo): {sil_full:.4f}")

    df_conn = df_conn.with_columns(
        pl.Series("cluster_id", final_labels.astype(np.int32))
    )

    # Figuras
    _plot_profiles(df_conn, FIGURES_DIR / FIG_PROFILES)
    _plot_razon_dist(df_conn, FIGURES_DIR / FIG_RAZON_DIST)

    _export_clusters(df_conn)

    # Unir al df completo; no conectados → -1
    df = df.join(
        df_conn.select(["call_url", "cluster_id"]),
        on="call_url",
        how="left",
    ).with_columns(pl.col("cluster_id").fill_null(-1).cast(pl.Int32))

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not CLEAN_CSV.exists():
        print(f"[ERROR] No se encontró: {CLEAN_CSV}")
        sys.exit(1)

    print(f"[INFO] Cargando {CLEAN_CSV.name}...")
    df_input  = pl.read_csv(CLEAN_CSV, infer_schema_length=5000)
    df_result = cluster_contacts(df_input)

    summary = (
        df_result.filter(pl.col("cluster_id") >= 0)
        .group_by("cluster_id")
        .agg(pl.len().alias("n"))
        .sort("cluster_id")
    )
    print("\nDistribución de clusters:")
    print(summary)
