"""
main.py - Punto de entrada del pipeline de analisis de llamadas.
Usa load_data para cargar el CSV y muestra los datos de forma agradable.
"""

import os
import sys
from pathlib import Path

# Forzar UTF-8 en Windows para evitar errores de codificacion
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Agregar scripts/ al path para importar módulos del pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from load_data import load_raw
from clean_data import clean


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
    import polars as pl
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

    # Agregar columnas (truncar transcript/post_call_analysis si existen)
    cols_anchas = {"transcript", "post_call_analysis", "call_url"}
    for col in df.columns:
        max_w = 30 if col in cols_anchas else None
        tabla.add_column(col, max_width=max_w, overflow="ellipsis")

    # Agregar filas
    for row in df.head(n).iter_rows():
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


def main():
    console.print()
    console.rule("[bold yellow]Pipeline de Analisis de Llamadas[/]")
    console.print()

    # Paso 1: Cargar datos
    console.print("[bold]>> Paso 1:[/] Cargando datos desde CSV...\n")
    df = load_raw()

    # Paso 2: Limpiar y normalizar
    console.print("\n[bold]>> Paso 2:[/] Limpiando y normalizando datos...\n")
    df_clean = clean(df)

    console.print()
    console.rule("[bold yellow]Visualizacion de Datos Limpios[/]")
    console.print()

    mostrar_resumen(df_clean)
    console.print()

    mostrar_columnas_nuevas(df_clean)
    console.print()

    mostrar_muestra(df_clean)
    console.print()

    console.rule("[bold green]Fase 2 completada exitosamente[/]")
    console.print()

    return df_clean


if __name__ == "__main__":
    main()
