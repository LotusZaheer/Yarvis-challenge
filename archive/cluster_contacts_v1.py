"""
Script 05 - Clustering de contactos conectados.

Entrada : pl.DataFrame con columnas del pipeline limpio.
Salida  : pl.DataFrame con columna cluster_id + CSV + figuras.

Sigue guia_clustering.md:
  - Features numéricas: duration_sec, hour (§9, §10)
  - Features binarias como numéricas: duration_outlier, inconsistency_flag (§2, §9)
  - Features categóricas: disconnected_reason, razon_grupo (macro), day_of_week (§3, §9)
  - Método principal: K-Prototypes con init Cao (§6)
  - Selección de K: muestra 15% + elbow + Silhouette (§7)
  - Baseline comparativo: K-Means con encoding ordinal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from utils.paths import CLEAN_CSV, PROCESSED_DIR, FIGURES_DIR
from utils.plotting import savefig, DPI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
K_RANGE      = range(2, 8)
SAMPLE_FRAC  = 0.15  # §7: muestra para selección de K

# Features (§9)
NUM_COLS = ["duration_sec", "hour"]
BIN_COLS = ["duration_outlier", "inconsistency_flag"]  # §2: binarias como numéricas
CAT_COLS = ["disconnected_reason", "razon_grupo", "day_of_week"]  # §9: las más informativas

# Parámetros de modelos
KPROTO_PARAMS = dict(init="Cao", n_init=1, max_iter=20, n_jobs=1)  # §6: init Cao
KMEANS_N_INIT = 10

# Figuras y salida
FIG_K_SELECTION   = "cluster_k_selection.png"
FIG_PROFILES      = "cluster_profiles.png"
FIG_RAZON_HEATMAP = "cluster_razon_heatmap.png"
CLUSTERS_CSV      = "clusters_contacts.csv"

# ---------------------------------------------------------------------------
# Preparación de variables (guia §3)
# ---------------------------------------------------------------------------

_RAZON_GRUPOS: dict[str, str] = {
    "falla_tecnica":      "tecnico",
    "fallas_servicio":    "tecnico",
    "dificultad_tecnica": "tecnico",
    "problema_tecnico":   "tecnico",
    "precio":             "precio",
    "precio_alto":        "precio",
    "costo":              "precio",
    "caro":               "precio",
    "competencia":        "competencia",
    "otro_operador":      "competencia",
    "portabilidad":       "competencia",
    "mala_atencion":      "experiencia",
    "mala_experiencia":   "experiencia",
    "insatisfaccion":     "experiencia",
    "olvido":             "operativos",
    "no_sabe_recargar":   "operativos",
    "sin_uso":            "operativos",
    "no_uso":             "operativos",
}


def _group_razon(val: str | None) -> str:
    """Asigna macro-grupo a pca_razon_churn; null -> sin_motivo."""
    if val is None:
        return "sin_motivo"
    return _RAZON_GRUPOS.get(str(val).strip().lower(), "otro")


def _prepare_variables(df: pl.DataFrame) -> pl.DataFrame:
    """
    Agrega columnas preparadas para clustering.
    §1: elimina features sparse (pca_sentimiento, pca_posible_recuperacion)
    §2: convierte binarias a Int32
    §3: agrupa pca_razon_churn en macro-grupos
    """
    return df.with_columns([
        # §3: macro-grupos de razón churn
        pl.Series("razon_grupo", [_group_razon(v) for v in df["pca_razon_churn"].to_list()]),
        # §2: binarias como numéricas 0/1
        pl.col("duration_outlier").cast(pl.Int32).alias("duration_outlier"),
        pl.col("inconsistency_flag").cast(pl.Int32).alias("inconsistency_flag"),
        # Asegurar disconnected_reason sin nulls
        pl.col("disconnected_reason").fill_null("unknown").alias("disconnected_reason"),
    ])


# ---------------------------------------------------------------------------
# Construcción de matrices (§8: preconvertir a NumPy una sola vez)
# ---------------------------------------------------------------------------

def _build_num_matrix(df: pl.DataFrame) -> np.ndarray:
    """Prepara parte numérica + binaria: fill_null(mediana) para continuas, 0 para binarias."""
    num_exprs = []
    for col in NUM_COLS:
        med = float(df[col].median() or 0.0)
        num_exprs.append(pl.col(col).fill_null(med).cast(pl.Float64))
    for col in BIN_COLS:
        num_exprs.append(pl.col(col).fill_null(0).cast(pl.Float64))
    return df.select(num_exprs).to_numpy()


def _build_cat_matrix(df: pl.DataFrame) -> np.ndarray:
    """Extrae la parte categórica como strings."""
    return df.select([pl.col(c).cast(pl.String) for c in CAT_COLS]).to_numpy()


def _build_kproto_matrix(df: pl.DataFrame):
    """
    Retorna (X_final, cat_indices, X_num_scaled).
      X_final     : numéricas escaladas | categóricas como str
      cat_indices : índices categóricos en X_final (para KPrototypes)
      X_num_scaled: solo parte numérica (para Silhouette aproximado)
    """
    X_num = StandardScaler().fit_transform(_build_num_matrix(df))
    X_cat = _build_cat_matrix(df)

    X_final     = np.hstack([X_num, X_cat])
    cat_indices = list(range(X_num.shape[1], X_final.shape[1]))
    return X_final, cat_indices, X_num


def _build_kmeans_matrix(df: pl.DataFrame) -> np.ndarray:
    """Encoding ordinal + escalado para K-Means baseline."""
    X_num     = _build_num_matrix(df)
    X_cat_raw = _build_cat_matrix(df)
    X_cat_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit_transform(X_cat_raw)

    return StandardScaler().fit_transform(np.hstack([X_num, X_cat_enc]))


# ---------------------------------------------------------------------------
# Selección de K (§5 elbow+silhouette, §7 con muestra)
# ---------------------------------------------------------------------------

def _sample_matrix(X, n_total: int, frac: float = SAMPLE_FRAC):
    """Genera índices de muestra estratificada para selección de K."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = max(int(n_total * frac), 500)  # mínimo 500 registros
    n_sample = min(n_sample, n_total)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    idx.sort()
    return idx


def _select_k(X_kp, cat_indices, X_num) -> tuple[int, dict, dict]:
    """Evalúa K-Prototypes sobre muestra para k en K_RANGE (§7)."""
    # §7: muestra para acelerar búsqueda de K
    idx = _sample_matrix(X_kp, X_kp.shape[0])
    X_kp_sample = X_kp[idx]
    X_num_sample = X_num[idx]
    print(f"[INFO] Muestra para selección de K: {len(idx):,} registros ({SAMPLE_FRAC:.0%} del total)")

    costs      = {}
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
            kp = KPrototypes(n_clusters=k, init="Cao", n_init=1,
                             random_state=RANDOM_STATE, n_jobs=-1)
            labels = kp.fit_predict(X_kp_sample, categorical=cat_indices)

            costs[k]      = float(kp.cost_)
            sil           = silhouette_score(X_num_sample, labels) if len(set(labels)) > 1 else -1.0
            sil_scores[k] = float(sil)

            prog.update(task, advance=1,
                        description=f"[yellow]K={k} | cost={kp.cost_:.0f} | sil={sil:.3f}[/]")

    best_k = max(sil_scores, key=sil_scores.get)
    return best_k, costs, sil_scores


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _plot_k_selection(costs: dict, sil_scores: dict, best_k: int, out_path: Path) -> None:
    """Figura doble: curva de costo (elbow) y Silhouette."""
    ks = sorted(costs.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, [costs[k] for k in ks], "o-", color="#1565C0", linewidth=2)
    ax1.axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    ax1.set_title("Elbow — Costo K-Prototypes", fontweight="bold")
    ax1.set_xlabel("Número de clusters k")
    ax1.set_ylabel("Costo total (disimilaridad)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ks, [sil_scores[k] for k in ks], "s-", color="#2E7D32", linewidth=2)
    ax2.axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    ax2.set_title("Silhouette Score (aprox. numérico)", fontweight="bold")
    ax2.set_xlabel("Número de clusters k")
    ax2.set_ylabel("Silhouette")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Selección del número óptimo de clusters", fontsize=13, fontweight="bold")
    savefig(fig, out_path, report=True)


def _cluster_cat_distribution(df_c: pl.DataFrame, col: str, cluster: int) -> dict[str, float]:
    """Calcula distribución porcentual de una columna categórica en un cluster."""
    sub   = df_c.filter(pl.col("cluster_id") == cluster)
    total = sub.height or 1
    return {
        val: sub.filter(pl.col(col) == val).height / total * 100
        for val in sorted(df_c[col].unique().to_list())
    }


def _plot_cluster_profiles(df_c: pl.DataFrame, out_path: Path) -> None:
    """Perfiles: stacked bar por categórica (arriba) + boxplots numéricos (abajo)."""
    clusters = sorted(df_c["cluster_id"].unique().to_list())
    n_cat    = len(CAT_COLS)
    n_num    = len(NUM_COLS)

    n_cols_grid = max(n_cat, n_num + 1)  # +1 para tabla de tamaños
    fig, axes = plt.subplots(2, n_cols_grid, figsize=(4 * n_cols_grid, 8))

    # Fila superior: stacked bar por categórica
    for col_idx, col in enumerate(CAT_COLS):
        ax   = axes[0, col_idx]
        vals = sorted(df_c[col].unique().to_list())
        colors = plt.cm.Set2(np.linspace(0, 0.9, len(vals)))
        bottom = np.zeros(len(clusters))

        for v_idx, val in enumerate(vals):
            heights = [
                _cluster_cat_distribution(df_c, col, cl).get(val, 0)
                for cl in clusters
            ]
            ax.bar(clusters, heights, bottom=bottom, label=str(val),
                   color=colors[v_idx], edgecolor="white", linewidth=0.5)
            bottom += np.array(heights)

        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("%" if col_idx == 0 else "")
        ax.set_xticks(clusters)
        ax.legend(fontsize=6, loc="upper right")

    for col_idx in range(n_cat, n_cols_grid):
        axes[0, col_idx].set_visible(False)

    # Fila inferior: boxplots por variable numérica + tabla de tamaños
    for num_idx, col in enumerate(NUM_COLS):
        ax = axes[1, num_idx]
        data = [df_c.filter(pl.col("cluster_id") == cl)[col].drop_nulls().to_list()
                for cl in clusters]
        ax.boxplot(data, labels=[f"C{c}" for c in clusters], patch_artist=True)
        ax.set_title(f"{col} por cluster", fontsize=9, fontweight="bold")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)

    # Tabla de tamaños
    ax_sz = axes[1, n_num]
    ax_sz.axis("off")
    sizes = df_c.group_by("cluster_id").agg(pl.len().alias("n")).sort("cluster_id")
    tdata = [[f"C{r['cluster_id']}", f"{r['n']:,}", f"{r['n']/df_c.height*100:.1f}%"]
             for r in sizes.iter_rows(named=True)]
    tbl   = ax_sz.table(cellText=tdata, colLabels=["Cluster", "N", "%"],
                        loc="center", cellLoc="center")
    tbl.scale(1.2, 1.5)
    ax_sz.set_title("Tamaño de clusters", fontsize=9, fontweight="bold")

    for col_idx in range(n_num + 1, n_cols_grid):
        axes[1, col_idx].set_visible(False)

    plt.suptitle("Perfiles de Clusters — Variables de Negocio", fontsize=13, fontweight="bold")
    savefig(fig, out_path, report=True)


def _plot_razon_heatmap(df_c: pl.DataFrame, out_path: Path) -> None:
    """Heatmap % razon_grupo x cluster."""
    clusters = sorted(df_c["cluster_id"].unique().to_list())
    grupos   = sorted(df_c["razon_grupo"].unique().to_list())

    matrix = np.array([
        [_cluster_cat_distribution(df_c, "razon_grupo", cl).get(g, 0) for cl in clusters]
        for g in grupos
    ])

    fig, ax = plt.subplots(figsize=(max(6, len(clusters) * 1.8), max(5, len(grupos) * 0.7)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="%")
    ax.set_xticks(range(len(clusters)))
    ax.set_yticks(range(len(grupos)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_yticklabels(grupos)

    for gi in range(len(grupos)):
        for ci in range(len(clusters)):
            v = matrix[gi, ci]
            ax.text(ci, gi, f"{v:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if v > 50 else "black")

    ax.set_title("Distribución de razon_grupo por cluster (%)", fontweight="bold")
    savefig(fig, out_path, report=True)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------

def _train_kprototypes(X_kp, cat_indices, best_k: int) -> np.ndarray:
    """Entrena K-Prototypes final sobre dataset completo."""
    print(f"[INFO] Entrenando K-Prototypes final (k={best_k})...")
    kp = KPrototypes(n_clusters=best_k, random_state=RANDOM_STATE, **KPROTO_PARAMS)
    return kp.fit_predict(X_kp, categorical=cat_indices)


def _run_kmeans_baseline(X_km, best_k: int) -> float:
    """Ejecuta K-Means baseline y retorna Silhouette score."""
    km     = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(X_km)
    return silhouette_score(X_km, labels) if len(set(labels)) > 1 else -1.0


def _export_clusters(df_prep: pl.DataFrame) -> None:
    """Exporta CSV de conectados con cluster_id."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / CLUSTERS_CSV
    df_prep.write_csv(out_path)
    n_clusters = df_prep["cluster_id"].n_unique()
    print(f"[INFO] Exportado: {out_path.name}  ({df_prep.height:,} registros, {n_clusters} clusters)")


# ---------------------------------------------------------------------------
# Función principal del pipeline
# ---------------------------------------------------------------------------

def cluster_contacts(df: pl.DataFrame) -> pl.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # §4: solo registros conectados
    df_conn = df.filter(pl.col("connected") == True)
    print(f"[INFO] Registros conectados: {df_conn.height:,}")

    df_prep = _prepare_variables(df_conn)

    # §8: construir matrices una sola vez
    X_kp, cat_indices, X_num = _build_kproto_matrix(df_prep)
    X_km                     = _build_kmeans_matrix(df_prep)

    # §7: seleccionar K con muestra + §5: elbow + silhouette
    print("[INFO] Evaluando K (elbow + silhouette sobre muestra)...")
    best_k, costs, sil_scores = _select_k(X_kp, cat_indices, X_num)
    print(f"[INFO] K sugerido: {best_k}  (sil={sil_scores[best_k]:.4f})")
    _plot_k_selection(costs, sil_scores, best_k, FIGURES_DIR / FIG_K_SELECTION)

    # §6: modelo final con init Cao sobre dataset completo
    final_labels = _train_kprototypes(X_kp, cat_indices, best_k)
    df_prep = df_prep.with_columns(
        pl.Series("cluster_id", final_labels.astype(np.int32))
    )

    # Silhouette final sobre dataset completo
    kp_sil_full = silhouette_score(X_num, final_labels) if len(set(final_labels)) > 1 else -1.0
    km_sil = _run_kmeans_baseline(X_km, best_k)
    print(f"[INFO] Silhouette — K-Prototypes: {kp_sil_full:.4f} | K-Means baseline: {km_sil:.4f}")

    # Figuras de perfiles
    _plot_cluster_profiles(df_prep, FIGURES_DIR / FIG_PROFILES)
    _plot_razon_heatmap(df_prep, FIGURES_DIR / FIG_RAZON_HEATMAP)

    _export_clusters(df_prep)

    # Unir cluster_id al df completo; no conectados -> -1
    df = df.join(
        df_prep.select(["call_url", "cluster_id"]),
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
