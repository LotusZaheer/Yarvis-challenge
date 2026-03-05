"""
Funciones de presentación y visualización compartidas entre main.py y main.ipynb.
Utiliza rich para renderizar tablas y paneles en consola.
"""

import polars as pl
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(force_terminal=True, safe_box=True)


def mostrar_resumen(df):
    """Muestra un resumen general del DataFrame en consola."""
    # --- Panel de información general ---
    info_lines = [
        f"[bold cyan]Registros totales:[/] {df.height:,}",
        f"[bold cyan]Columnas:[/]         {len(df.columns)}",
        f"[bold cyan]Memoria:[/]          {df.estimated_size() / 1024**2:.2f} MB",
    ]
    console.print(Panel(
        "\n".join(info_lines),
        title="[bold green]Resumen del Dataset[/]",
        border_style="green",
        padding=(1, 2),
    ))

    # --- Tabla de tipos de dato y nulos ---
    tabla_tipos = Table(
        title="Tipos de Dato y Valores Nulos",
        box=box.ROUNDED,
        header_style="bold magenta",
        show_lines=True,
    )
    tabla_tipos.add_column("Columna", style="cyan", min_width=20)
    tabla_tipos.add_column("Tipo", style="yellow", justify="center")
    tabla_tipos.add_column("Nulos", style="red", justify="right")
    tabla_tipos.add_column("% Nulos", style="red", justify="right")
    tabla_tipos.add_column("Válidos", style="green", justify="right")
    tabla_tipos.add_column("% Válidos", style="green", justify="right")
    tabla_tipos.add_column("Valores Distintos", style="cyan", justify="right")

    for col in df.columns:
        nulos = df[col].null_count()
        validos = df.height - nulos
        pct_nulos = (nulos / df.height) * 100
        pct_validos = 100 - pct_nulos
        distintos = df[col].n_unique()
        tabla_tipos.add_row(
            col,
            str(df[col].dtype),
            f"{nulos:,}",
            f"{pct_nulos:.1f}%",
            f"{validos:,}",
            f"{pct_validos:.1f}%",
            f"{distintos:,}"
        )

    console.print(tabla_tipos)

    # --- Estadísticas de conexión ---
    if "connected" in df.columns:
        conectadas = df.filter(pl.col("connected") == True).height
        no_conectadas = df.height - conectadas
        console.print(Panel(
            f"[bold green]Conectadas:[/]    {conectadas:,}  ({conectadas/df.height*100:.1f}%)\n"
            f"[bold red]No conectadas:[/] {no_conectadas:,}  ({no_conectadas/df.height*100:.1f}%)",
            title="[bold blue]Estado de Conexion[/]",
            border_style="blue",
            padding=(1, 2),
        ))


def mostrar_muestra(df, n=5):
    """Muestra las primeras n filas del DataFrame como tabla rich."""
    tabla = Table(
        title=f"Primeras {n} Filas del Dataset",
        box=box.HEAVY_HEAD,
        header_style="bold white on dark_green",
        row_styles=["", "dim"],
        show_lines=False,
    )

    # Mostrar columnas relevantes (excluye columnas de texto muy largo)
    cols_anchas = {"transcript", "post_call_analysis", "call_url"}
    view_cols = ['campaign_id', 'target_id',
                 'connected', 'call_completed', 'disconnected_reason',
                 'executed_at', 'hour', 'day_of_week', 'date', 'duration_sec', 'duration_outlier', 'inconsistency_flag']
    # Solo incluir columnas que existan en el df
    view_cols = [c for c in view_cols if c in df.columns]

    for col in view_cols:
        max_w = 30 if col in cols_anchas else None
        tabla.add_column(col, max_width=max_w, overflow="ellipsis")

    # Agregar filas
    for row in df.select(view_cols).head(n).iter_rows():
        tabla.add_row(*[str(v) if v is not None and str(v) != "nan" else "—" for v in row])

    console.print(tabla)


def mostrar_columnas_nuevas(df):
    """Muestra solo las columnas agregadas en la Fase 2."""
    nuevas = [
        "hour", "day_of_week", "date",
        "duration_sec", "duration_outlier",
        "transcript_length", "inconsistency_flag",
        "pca_sentimiento", "pca_razon_churn", "pca_posible_recuperacion",
    ]
    cols_presentes = [c for c in nuevas if c in df.columns]
    if not cols_presentes:
        return

    tabla = Table(
        title="Columnas Nuevas - Fase 2 (primeras 5 filas)",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    for col in cols_presentes:
        tabla.add_column(col, max_width=25, overflow="ellipsis")

    for row in df.select(cols_presentes).head(5).iter_rows():
        tabla.add_row(*[str(v) if v is not None else "—" for v in row])

    console.print(tabla)
