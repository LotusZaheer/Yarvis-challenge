"""
Script 05 - Clustering de contactos conectados (Tarea 1).
Entrada : pl.DataFrame con sentiment_own (salida de sentiment_analysis.py)
Salida  : pl.DataFrame con columna cluster_id + data/processed/clusters_contacts.csv
          + figuras en reports/figures/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "calls_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
K_RANGE = range(2, 9)


def _build_features(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Construye matriz de features para clustering."""
    feature_cols = []

    # Numéricas
    df = df.with_columns([
        pl.col("duration_sec").fill_null(pl.col("duration_sec").mean()),
        pl.col("transcript_length").fill_null(0.0).cast(pl.Float64),
        pl.col("hour").cast(pl.Float64),
    ])
    feature_cols += ["duration_sec", "transcript_length", "hour"]

    # One-hot: razón de desconexión (categorías relevantes en llamadas conectadas)
    for reason in ["user_hangup", "agent_hangup", "inactivity"]:
        col = f"dr_{reason}"
        df = df.with_columns(
            (pl.col("disconnected_reason") == reason).cast(pl.Float64).alias(col)
        )
        feature_cols.append(col)

    # One-hot: sentiment_own (si existe)
    if "sentiment_own" in df.columns:
        for sent in ["positivo", "negativo"]:
            col = f"sent_{sent}"
            df = df.with_columns(
                (pl.col("sentiment_own") == sent).cast(pl.Float64).alias(col)
            )
            feature_cols.append(col)

    X = df.select(feature_cols).to_numpy().astype(float)
    # Imputar NaN residuales con 0
    X = np.nan_to_num(X, nan=0.0)
    return X, feature_cols


def _plot_elbow_silhouette(inertias: dict, silhouettes: dict, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ks = sorted(inertias.keys())

    ax1.plot(ks, [inertias[k] for k in ks], "o-", color="#2196F3")
    ax1.set_xlabel("Número de clusters k")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Método del Codo", fontweight="bold")
    ax1.set_xticks(ks)

    ax2.plot(ks, [silhouettes[k] for k in ks], "o-", color="#4CAF50")
    best_k = max(silhouettes, key=silhouettes.get)
    ax2.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"k óptimo = {best_k}")
    ax2.set_xlabel("Número de clusters k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score por k", fontweight="bold")
    ax2.set_xticks(ks)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cluster_profiles(df_connected: pl.DataFrame, out_path: Path):
    """Gráfico de perfil por cluster: duración media y longitud de transcript."""
    profile = (
        df_connected.group_by("cluster_id")
        .agg([
            pl.col("duration_sec").mean().alias("dur_media"),
            pl.col("transcript_length").mean().alias("trans_media"),
            pl.len().alias("n"),
        ])
        .sort("cluster_id")
    )

    clusters = profile["cluster_id"].to_list()
    dur = profile["dur_media"].to_list()
    trans = profile["trans_media"].to_list()

    x = np.arange(len(clusters))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width / 2, dur, width, label="Duración media (s)", color="#2196F3", alpha=0.8)
    b2 = ax2.bar(x + width / 2, trans, width, label="Largo transcript (chars)", color="#FF9800", alpha=0.8)

    ax1.set_xlabel("Cluster ID")
    ax1.set_ylabel("Duración media (segundos)", color="#2196F3")
    ax2.set_ylabel("Longitud transcript media (chars)", color="#FF9800")
    ax1.set_title("Perfil de Clusters: Duración y Transcript", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Cluster {c}" for c in clusters])

    lines = [b1, b2]
    labels = [b.get_label() for b in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cluster_contacts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica KMeans sobre registros conectados.
    Si clusters_contacts.csv ya existe, carga cluster_id desde ahí (join por call_url).
    Retorna df completo con columna cluster_id (-1 para no-conectados).
    Exporta data/processed/clusters_contacts.csv con los registros conectados.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    _clusters_csv = PROCESSED_DIR / "clusters_contacts.csv"
    _figures = [FIGURES_DIR / "cluster_elbow_silhouette.png", FIGURES_DIR / "cluster_profiles.png"]
    _MIN_SIZE_KB = 5

    if _clusters_csv.exists():
        cached = pl.read_csv(_clusters_csv).select(["call_url", "cluster_id"])
        # Validar que el CSV tiene las filas esperadas
        if cached.height == df.height:
            df = df.join(cached, on="call_url", how="left")
            df = df.with_columns(pl.col("cluster_id").fill_null(-1).cast(pl.Int32))
            if not all(f.exists() and f.stat().st_size > _MIN_SIZE_KB * 1024 for f in _figures):
                # Regenerar figuras si faltan, usando los datos del cache
                df_connected = df.filter(pl.col("connected") == True)
                _plot_cluster_profiles(df_connected, FIGURES_DIR / "cluster_profiles.png")
            print(f"[INFO] Cache encontrado: {_clusters_csv.name} ({cached.height:,} filas)")
            return df
        print(f"[WARN] Cache invalido (filas: {cached.height:,} vs esperadas: {df.height:,}), recalculando...")

    df_connected = df.filter(pl.col("connected") == True)
    print(f"[INFO] Clustering sobre {df_connected.height:,} llamadas conectadas")

    X, feature_cols = _build_features(df_connected)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Selección de k óptimo por silhouette
    inertias = {}
    silhouettes = {}
    sample_size = min(5000, X_scaled.shape[0])
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = float(km.inertia_)
        silhouettes[k] = float(
            silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE)
        )

    best_k = max(silhouettes, key=silhouettes.get)
    print(f"[INFO] k optimo por silhouette: {best_k} (score={silhouettes[best_k]:.4f})")

    _plot_elbow_silhouette(inertias, silhouettes, FIGURES_DIR / "cluster_elbow_silhouette.png")

    # Modelo final
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = km_final.fit_predict(X_scaled)

    df_connected = df_connected.with_columns(
        pl.Series("cluster_id", cluster_ids.tolist(), dtype=pl.Int32)
    )

    _plot_cluster_profiles(df_connected, FIGURES_DIR / "cluster_profiles.png")

    # Exportar CSV de clústeres (call_url incluido como clave de cache)
    export_cols = ["call_url", "campaign_id", "target_id", "cluster_id", "duration_sec",
                   "transcript_length", "hour", "day_of_week", "disconnected_reason"]
    if "sentiment_own" in df_connected.columns:
        export_cols.append("sentiment_own")
    if "campaign_type" in df_connected.columns:
        export_cols.append("campaign_type")

    out_cols = [c for c in export_cols if c in df_connected.columns]
    out_path = PROCESSED_DIR / "clusters_contacts.csv"
    df_connected.select(out_cols).write_csv(out_path)
    print(f"[INFO] Exportado: clusters_contacts.csv ({df_connected.height:,} filas, k={best_k})")

    # Agregar cluster_id al df completo (-1 para no conectados, join vectorizado por call_url)
    df = (
        df.join(
            df_connected.select(["call_url", "cluster_id"]),
            on="call_url",
            how="left",
        )
        .with_columns(pl.col("cluster_id").fill_null(-1).cast(pl.Int32))
    )

    return df


if __name__ == "__main__":
    df = pl.read_csv(CLEAN_CSV)
    df = cluster_contacts(df)
    print(df.filter(pl.col("cluster_id") >= 0).select(["target_id", "cluster_id"]).head(5))
