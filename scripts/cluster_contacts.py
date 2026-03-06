"""
Script 05c — Clustering de contactos conectados (v3).

Correcciones respecto a v1/v2:
  - Sin filtro de nulls: usa los 8,855 conectados completos (v2 dejaba solo 539)
  - pca_sentimiento imputado como neutral en vez de excluir filas
  - pca_razon_churn → tiene_motivo binaria (0/1) — no excluye por null
  - day_of_week descartado (Cramér's V < 0.09 con todo → ruido)
  - n_jobs=1 explícito + vars de entorno OMP/LOKY para evitar deadlocks en Windows
  - Heartbeat cada 8s + timeout de 120s por K para detectar bucles infinitos
  - silhouette_score con sample_size para rapidez en datasets grandes
"""

# OMP_NUM_THREADS y LOKY_MAX_CPU_COUNT deben setearse ANTES de importar numpy/sklearn
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"]    = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]    = "1"

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.df_helpers import connected_calls
from utils.paths import CLEAN_CSV, PROCESSED_DIR, FIGURES_DIR, ensure_output_dirs
from utils.plotting import savefig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RANDOM_STATE   = 42
K_RANGE        = range(2, 7)   # 2, 3, 4, 5, 6
N_INIT         = 10
MAX_ITER       = 150           # suficiente para convergencia; evita esperas largas
SAMPLE_N       = 2_000         # registros para selección de K (de 8855 total)
SIL_SAMPLE_K   = 1_000         # muestra para silhouette durante selección K
SIL_SAMPLE_FIN = 3_000         # muestra para silhouette final
TIMEOUT_K_SEC  = 120           # segundos máx por K antes de reportar timeout
HEARTBEAT_SECS = 8             # frecuencia del heartbeat de vida

# Sobreescritura manual: si no es None, ignora la selección automática de K.
# Útil cuando el silhouette favorece K=2 por dominio de disconnected_reason
# pero se quiere más granularidad (e.g., FORCE_K = 3 ó 4).
FORCE_K: int | None = 4

# Encoding de dominio para categóricas
ENGAGEMENT_MAP = {"user_hangup": 2, "agent_hangup": 1}   # resto → 0
SENTIMENT_MAP  = {"positivo": 2, "neutral": 1, "negativo": 0}  # null → 1

# Valores que indican "sin motivo churn"
SIN_MOTIVO = {None, "", "n/a", "otro", "sin_motivo"}

# Salidas
FIG_K_SEL    = "cluster_k_selection.png"
FIG_PROFILES = "cluster_profiles.png"
FIG_RAZON    = "cluster_razon_dist.png"
OUT_CSV      = "clusters_contacts.csv"


# ---------------------------------------------------------------------------
# Heartbeat — detecta si el proceso está colgado o avanzando
# ---------------------------------------------------------------------------

class _Heartbeat(threading.Thread):
    """
    Hilo daemon que imprime el tiempo transcurrido cada HEARTBEAT_SECS segundos.
    Si deja de imprimir, el proceso principal se colgó.
    Uso: with _Heartbeat("label"):  ...código...
    """

    def __init__(self, label: str, interval: float = HEARTBEAT_SECS):
        super().__init__(daemon=True)
        self._stop_evt  = threading.Event()
        self._label     = label
        self._interval  = interval
        self._t0        = time.time()

    def run(self) -> None:
        while not self._stop_evt.wait(self._interval):
            elapsed = time.time() - self._t0
            print(f"  [ALIVE] {self._label} — {elapsed:.0f}s transcurridos...", flush=True)

    def stop(self) -> None:
        self._stop_evt.set()

    # Soporte como context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Encoding de features (sin filtrar nulls)
# ---------------------------------------------------------------------------

def _encode_features(df: pl.DataFrame) -> np.ndarray:
    """
    Convierte las 7 features a float64.
    Nunca excluye filas — imputa nulls con valores de dominio.
    """
    dur_med = float(df["duration_sec"].median() or 0.0)

    # 1. duration_sec — imputa con mediana del mismo df
    duration = df["duration_sec"].fill_null(dur_med).to_numpy(allow_copy=True).astype(float)

    # 2. hour
    hour = df["hour"].to_numpy(allow_copy=True).astype(float)

    # 3. disconnected_reason → jerarquía engagement (null → 0)
    engagement = np.array(
        [ENGAGEMENT_MAP.get(str(v) if v is not None else "", 0)
         for v in df["disconnected_reason"].to_list()],
        dtype=float,
    )

    # 4. duration_outlier 0/1
    outlier = df["duration_outlier"].cast(pl.Float64).fill_null(0).to_numpy(allow_copy=True)

    # 5. inconsistency_flag 0/1
    inconsistency = df["inconsistency_flag"].cast(pl.Float64).fill_null(0).to_numpy(allow_copy=True)

    # 6. pca_sentimiento → ordinal (null → 1 = neutral)
    sentiment = np.array(
        [SENTIMENT_MAP.get(str(v) if v is not None else "", 1)
         for v in df["pca_sentimiento"].to_list()],
        dtype=float,
    )

    # 7. tiene_motivo churn → 0 si null/"n/a"/"otro", else 1
    tiene_motivo = np.array(
        [0.0 if v in SIN_MOTIVO else 1.0
         for v in df["pca_razon_churn"].to_list()],
        dtype=float,
    )

    return np.column_stack([duration, hour, engagement, outlier, inconsistency, sentiment, tiene_motivo])


# ---------------------------------------------------------------------------
# Selección de K con heartbeat y timeout
# ---------------------------------------------------------------------------

def _fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Crea y ajusta KMeans con parámetros seguros para Windows."""
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        # n_jobs eliminado en sklearn>=1.2; OMP_NUM_THREADS=1 al inicio del módulo controla threads
        algorithm="lloyd",
    )
    km.fit(X)
    return km


def _select_k(X_scaled: np.ndarray) -> tuple[int, dict, dict]:
    """
    Evalúa K en K_RANGE sobre una muestra de SAMPLE_N registros.
    Cada K tiene un timeout de TIMEOUT_K_SEC segundos.
    Imprime heartbeat y timing por K para detectar bucles infinitos.
    """
    rng      = np.random.default_rng(RANDOM_STATE)
    n_sample = min(SAMPLE_N, X_scaled.shape[0])
    idx      = rng.choice(X_scaled.shape[0], size=n_sample, replace=False)
    X_s      = X_scaled[idx]

    print(f"[INFO] Muestra selección K: {n_sample:,} registros de {X_scaled.shape[0]:,}")
    print(f"[INFO] K_RANGE={list(K_RANGE)}, n_init={N_INIT}, max_iter={MAX_ITER}, "
          f"timeout_por_k={TIMEOUT_K_SEC}s, heartbeat_cada={HEARTBEAT_SECS}s")

    inertias   = {}
    sil_scores = {}
    timed_out  = []

    for k in K_RANGE:
        t_start = time.time()
        print(f"\n[K={k}] Iniciando KMeans... ", end="", flush=True)

        with _Heartbeat(f"KMeans k={k}"):
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_fit_kmeans, X_s, k)
                try:
                    km = future.result(timeout=TIMEOUT_K_SEC)
                except FuturesTimeout:
                    elapsed = time.time() - t_start
                    print(f"\n[TIMEOUT] K={k} superó {elapsed:.0f}s — se omite este K.", flush=True)
                    timed_out.append(k)
                    continue

        elapsed = time.time() - t_start
        labels  = km.labels_

        sil = -1.0
        if len(set(labels)) > 1:
            sil = float(silhouette_score(
                X_s, labels,
                sample_size=min(SIL_SAMPLE_K, len(labels)),
                random_state=RANDOM_STATE,
            ))

        inertias[k]   = float(km.inertia_)
        sil_scores[k] = sil
        print(f"listo en {elapsed:.1f}s | inertia={km.inertia_:.0f} | sil={sil:.4f}", flush=True)

    if not sil_scores:
        raise RuntimeError(
            "Todos los valores de K agotaron el timeout. "
            "Reducir N_INIT o MAX_ITER en la configuración."
        )

    if timed_out:
        print(f"\n[WARN] K con timeout (omitidos): {timed_out}")

    best_k = max(sil_scores, key=sil_scores.get)
    return best_k, inertias, sil_scores


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _plot_k_selection(inertias: dict, sil_scores: dict, best_k: int, out: Path) -> None:
    ks  = sorted(inertias.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, [inertias[k] for k in ks], "o-", color="#1565C0", linewidth=2)
    ax1.axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    ax1.set_title("Elbow — Inercia K-Means", fontweight="bold")
    ax1.set_xlabel("Número de clusters k"); ax1.set_ylabel("Inercia (SSE)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ks, [sil_scores[k] for k in ks], "s-", color="#2E7D32", linewidth=2)
    ax2.axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("Número de clusters k"); ax2.set_ylabel("Silhouette")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Selección K óptimo (K-Means, todos los conectados)", fontsize=13, fontweight="bold")
    savefig(fig, out, report=True)


def _pct(df_c: pl.DataFrame, col: str, val, cluster: int) -> float:
    sub   = df_c.filter(pl.col("cluster_id") == cluster)
    total = sub.height or 1
    return sub.filter(pl.col(col) == val).height / total * 100


def _plot_profiles(df_c: pl.DataFrame, out: Path) -> None:
    clusters = sorted(df_c["cluster_id"].unique().to_list())
    n_cl     = len(clusters)
    colors   = plt.cm.Set2(np.linspace(0, 0.8, n_cl))

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))

    # Fila 0: boxplots numéricos
    for ax_i, col in enumerate(["duration_sec", "hour"]):
        ax   = axes[0, ax_i]
        data = [df_c.filter(pl.col("cluster_id") == cl)[col].drop_nulls().to_list()
                for cl in clusters]
        bp = ax.boxplot(data, tick_labels=[f"C{c}" for c in clusters],
                        patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_ylabel(col); ax.grid(axis="y", alpha=0.3)

    # Tabla de tamaños
    ax_sz = axes[0, 2]
    ax_sz.axis("off")
    sizes = df_c.group_by("cluster_id").agg(pl.len().alias("n")).sort("cluster_id")
    tdata = [[f"C{r['cluster_id']}", f"{r['n']:,}", f"{r['n']/df_c.height*100:.1f}%"]
             for r in sizes.iter_rows(named=True)]
    tbl = ax_sz.table(cellText=tdata, colLabels=["Cluster", "N", "%"],
                      loc="center", cellLoc="center")
    tbl.scale(1.2, 1.8)
    ax_sz.set_title("Tamaño de clusters", fontsize=10, fontweight="bold")
    axes[0, 3].set_visible(False)

    # Fila 1: stacked bars categóricas
    cat_specs = [
        ("disconnected_reason", "Razón desconexión"),
        ("pca_sentimiento",     "Sentimiento"),
        ("tiene_motivo_str",    "Motivo churn"),
    ]
    df_c = df_c.with_columns(
        pl.when(pl.col("pca_razon_churn").is_null() |
                pl.col("pca_razon_churn").is_in(["n/a", "otro", "", "sin_motivo"]))
        .then(pl.lit("sin_motivo"))
        .otherwise(pl.lit("con_motivo"))
        .alias("tiene_motivo_str")
    )
    for ax_i, (col, title) in enumerate(cat_specs):
        ax   = axes[1, ax_i]
        vals = sorted(df_c[col].drop_nulls().unique().to_list())
        cmap = plt.cm.Set3(np.linspace(0, 1, max(len(vals), 1)))
        bottom = np.zeros(n_cl)
        for vi, val in enumerate(vals):
            heights = [_pct(df_c, col, val, cl) for cl in clusters]
            ax.bar(clusters, heights, bottom=bottom, label=str(val),
                   color=cmap[vi], edgecolor="white", linewidth=0.5)
            bottom += np.array(heights)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("%" if ax_i == 0 else "")
        ax.set_xticks(clusters)
        ax.legend(fontsize=7, loc="upper right")

    axes[1, 3].set_visible(False)
    plt.suptitle("Perfiles de Clusters — Variables de Negocio", fontsize=13, fontweight="bold")
    savefig(fig, out, report=True)


def _plot_razon_dist(df_c: pl.DataFrame, out: Path) -> None:
    clusters = sorted(df_c["cluster_id"].unique().to_list())
    n_cl     = len(clusters)

    razones = (
        df_c.filter(pl.col("pca_razon_churn").is_not_null())
            .filter(~pl.col("pca_razon_churn").is_in(["n/a", "otro", "", "sin_motivo"]))
            ["pca_razon_churn"].value_counts()
            .sort("count", descending=True)
            .head(10)["pca_razon_churn"].to_list()
    )
    if not razones:
        print("[WARN] Sin datos suficientes para cluster_v3_razon_dist.png")
        return

    matrix = np.array([
        [_pct(df_c, "pca_razon_churn", r, cl) for cl in clusters]
        for r in razones
    ])

    fig, ax = plt.subplots(figsize=(max(6, n_cl * 2), max(5, len(razones) * 0.7)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="%")
    ax.set_xticks(range(n_cl)); ax.set_yticks(range(len(razones)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_yticklabels(razones)
    for gi in range(len(razones)):
        for ci in range(n_cl):
            v = matrix[gi, ci]
            ax.text(ci, gi, f"{v:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if v > 50 else "black")
    ax.set_title("Distribución razón churn por cluster (%)", fontweight="bold")
    savefig(fig, out, report=True)


# ---------------------------------------------------------------------------
# Entrenamiento final y exportación
# ---------------------------------------------------------------------------

def _train_final(X_scaled: np.ndarray, best_k: int) -> np.ndarray:
    """Entrena KMeans final sobre todos los registros con heartbeat."""
    print(f"\n[INFO] Entrenando KMeans final (k={best_k}, {X_scaled.shape[0]:,} registros)...")
    t0 = time.time()
    with _Heartbeat(f"KMeans final k={best_k}"):
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_fit_kmeans, X_scaled, best_k)
            try:
                km = future.result(timeout=TIMEOUT_K_SEC * 2)
            except FuturesTimeout:
                raise RuntimeError(
                    f"KMeans final (k={best_k}) superó {TIMEOUT_K_SEC*2}s. "
                    "Reducir N_INIT o MAX_ITER."
                )
    print(f"[INFO] KMeans final listo en {time.time()-t0:.1f}s", flush=True)
    return km.labels_


def _export_clusters(df_c: pl.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / OUT_CSV
    df_c.write_csv(out)
    print(f"[INFO] Exportado: {out.name}  ({df_c.height:,} registros, {df_c['cluster_id'].n_unique()} clusters)")


# ---------------------------------------------------------------------------
# Función principal del pipeline
# ---------------------------------------------------------------------------

def cluster_contacts(df: pl.DataFrame) -> pl.DataFrame:
    ensure_output_dirs()
    t_total = time.time()

    # Solo conectados — SIN filtrar nulls en features
    df_conn = connected_calls(df)
    print(f"[INFO] Registros conectados: {df_conn.height:,}  (todos se usan para clustering)")

    # Encoding + escalado sobre todos los conectados
    print("[INFO] Codificando features...")
    X_raw  = _encode_features(df_conn)
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)
    print(f"[INFO] Matriz de features: {X.shape[0]:,} × {X.shape[1]}  (duration_sec, hour, "
          "engagement, outlier, inconsistency, sentiment, tiene_motivo)")

    # Selección de K
    print("\n[INFO] Evaluando K (elbow + silhouette sobre muestra)...")
    best_k, inertias, sil_scores = _select_k(X)
    print(f"\n[INFO] K sugerido por silhouette: {best_k}  (sil={sil_scores[best_k]:.4f})")
    if FORCE_K is not None:
        best_k = FORCE_K
        print(f"[INFO] K sobreescrito por FORCE_K={FORCE_K}")
    _plot_k_selection(inertias, sil_scores, best_k, FIGURES_DIR / FIG_K_SEL)

    # Modelo final
    final_labels = _train_final(X, best_k)

    # Silhouette final (con muestra para rapidez)
    sil_final = float(silhouette_score(
        X, final_labels,
        sample_size=min(SIL_SAMPLE_FIN, X.shape[0]),
        random_state=RANDOM_STATE,
    ))
    print(f"[INFO] Silhouette final: {sil_final:.4f}")

    # Adjuntar cluster_id al df conectado
    df_conn = df_conn.with_columns(
        pl.Series("cluster_id", final_labels.astype(np.int32))
    )

    # Figuras
    _plot_profiles(df_conn, FIGURES_DIR / FIG_PROFILES)
    _plot_razon_dist(df_conn, FIGURES_DIR / FIG_RAZON)

    _export_clusters(df_conn)

    # Join al df completo; no conectados → -1
    df = df.join(
        df_conn.select(["call_url", "cluster_id"]),
        on="call_url",
        how="left",
    ).with_columns(pl.col("cluster_id").fill_null(-1).cast(pl.Int32))

    print(f"\n[INFO] cluster_contacts completado en {time.time()-t_total:.1f}s total")
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
