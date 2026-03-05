"""
Script 05 - Clustering de contactos conectados (SOTA: K-Prototypes).
Entrada : pl.DataFrame con sentiment_own.
Salida  : pl.DataFrame con columna cluster_id + exportación CSV y figuras.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "calls_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
K_RANGE = range(2, 9)

# --- FUNCIONES DE APOYO ---

def _prepare_data_for_kproto(df: pl.DataFrame):
    """
    Prepara matriz mixta: numéricas escaladas + categóricas como strings.
    Evita One-Hot encoding para mejorar interpretabilidad y velocidad.
    """
    # 1. Definir tipos de columnas
    num_cols = ["duration_sec", "transcript_length", "hour"]
    cat_cols = ["disconnected_reason"]
    if "sentiment_own" in df.columns:
        cat_cols.append("sentiment_own")

    # 2. Procesar numéricas: Imputación rápida y escalado
    df_num = df.select([
        pl.col("duration_sec").fill_null(pl.col("duration_sec").mean()),
        pl.col("transcript_length").fill_null(0.0).cast(pl.Float64),
        pl.col("hour").cast(pl.Float64),
    ])
    
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_num.to_numpy())

    # 3. Procesar categóricas: Llenar nulos como categoría 'unknown'
    df_cat = df.select([
        pl.col(c).fill_null("unknown").cast(pl.String) for c in cat_cols
    ])
    X_cat = df_cat.to_numpy()

    # Combinamos matrices (Numéricas primero, Categóricas después)
    X_final = np.hstack([X_num, X_cat])
    
    # Identificar índices de las columnas categóricas para K-Prototypes
    cat_indices = list(range(X_num.shape[1], X_final.shape[1]))
    
    return X_final, cat_indices, X_num

def _plot_evaluation_metrics(scores: dict, out_path: Path):
    """Gráfico de métrica Calinski-Harabasz para selección de K."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    
    ax.plot(ks, vals, "o-", color="#673AB7", linewidth=2)
    best_k = max(scores, key=scores.get)
    ax.axvline(best_k, color="red", linestyle="--", label=f"K sugerido = {best_k}")
    
    ax.set_title("Calinski-Harabasz Index (Higher is better)", fontweight="bold")
    ax.set_xlabel("Número de clusters k")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_cluster_profiles(df_connected: pl.DataFrame, out_path: Path):
    """Perfilado de clusters mediante agregación de Polars."""
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
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - width/2, dur, width, label="Duración (s)", color="#2196F3")
    ax2.bar(x + width/2, trans, width, label="Transcript (chars)", color="#FF9800")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Cluster {c}\n(n={n})" for c, n in zip(clusters, profile["n"].to_list())])
    ax1.set_title("Perfil de Clusters: Métricas de Negocio", fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# --- FUNCIÓN PRINCIPAL ---

def cluster_contacts(df: pl.DataFrame) -> pl.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_connected = df.filter(pl.col("connected") == True)
    print(f"[INFO] Iniciando clustering SOTA sobre {df_connected.height:,} registros conectados")

    # 1. Preparación de datos
    X, cat_indices, X_num_only = _prepare_data_for_kproto(df_connected)
    
    # 2. Búsqueda de K óptimo con Calinski-Harabasz (Métrica O(n))
    scores = {}
    best_k = 2
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("[yellow]Optimizando K (K-Prototypes)...[/]", total=len(K_RANGE))
        
        for k in K_RANGE:
            # Cao es un método de inicialización más rápido para datos categóricos
            kp = KPrototypes(n_clusters=k, init='Cao', n_init=1, random_state=RANDOM_STATE, n_jobs=-1)
            labels = kp.fit_predict(X, categorical=cat_indices)
            
            # El score se calcula sobre la parte numérica
            score = calinski_harabasz_score(X_num_only, labels)
            scores[k] = score
            
            progress.update(task, advance=1, description=f"[yellow]K={k} | Score={score:.2f}[/]")

    best_k = max(scores, key=scores.get)
    print(f"[INFO] K sugerido: {best_k} (Score: {scores[best_k]:.2f})")
    _plot_evaluation_metrics(scores, FIGURES_DIR / "cluster_evaluation.png")

    # 3. Entrenamiento Final
    print(f"[INFO] Entrenando modelo final con K-Prototypes (n_init=3)...")
    kp_final = KPrototypes(n_clusters=best_k, init='Cao', n_init=3, random_state=RANDOM_STATE, n_jobs=-1)
    final_labels = kp_final.fit_predict(X, categorical=cat_indices)

    # 4. Integración y Exportación
    df_connected = df_connected.with_columns(
        pl.Series("cluster_id", final_labels.astype(np.int32))
    )

    _plot_cluster_profiles(df_connected, FIGURES_DIR / "cluster_profiles.png")

    # Exportar solo registros conectados
    out_path = PROCESSED_DIR / "clusters_contacts.csv"
    df_connected.write_csv(out_path)
    print(f"[INFO] ✓ Exportado: {out_path.name}")

    # Unir al dataframe original
    df = df.join(
        df_connected.select(["call_url", "cluster_id"]),
        on="call_url",
        how="left"
    ).with_columns(pl.col("cluster_id").fill_null(-1).cast(pl.Int32))

    print(f"[INFO] Proceso finalizado. {df_connected.height:,} llamadas asignadas a {best_k} clusters.")
    return df

if __name__ == "__main__":
    if CLEAN_CSV.exists():
        df_input = pl.read_csv(CLEAN_CSV)
        df_result = cluster_contacts(df_input)
        print("\nPrimeras filas con clusters:")
        print(df_result.filter(pl.col("cluster_id") >= 0).select(["target_id", "cluster_id"]).head(5))
    else:
        print(f"[ERROR] No se encontró el archivo {CLEAN_CSV}")