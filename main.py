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


console = Console(force_terminal=True, safe_box=True)


def mostrar_resumen(df):
    """Muestra un resumen general del DataFrame en consola."""
    # --- Panel de información general ---
    info_lines = [
        f"[bold cyan]Registros totales:[/] {len(df):,}",
        f"[bold cyan]Columnas:[/]         {len(df.columns)}",
        f"[bold cyan]Memoria:[/]          {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
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

    for col in df.columns:
        nulos = df[col].isnull().sum()
        pct = f"{nulos / len(df) * 100:.1f}%"
        tabla_tipos.add_row(col, str(df[col].dtype), f"{nulos:,}", pct)

    console.print(tabla_tipos)

    # --- Estadísticas de conexión ---
    if "connected" in df.columns:
        conectadas = (df["connected"] == True).sum()  # noqa: E712
        no_conectadas = len(df) - conectadas
        console.print(Panel(
            f"[bold green]Conectadas:[/]    {conectadas:,}  ({conectadas/len(df)*100:.1f}%)\n"
            f"[bold red]No conectadas:[/] {no_conectadas:,}  ({no_conectadas/len(df)*100:.1f}%)",
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
    for _, row in df.head(n).iterrows():
        tabla.add_row(*[str(v) if str(v) != "nan" else "—" for v in row])

    console.print(tabla)


def main():
    console.print()
    console.rule("[bold yellow]Pipeline de Analisis de Llamadas[/]")
    console.print()

    # Paso 1: Cargar datos
    console.print("[bold]>> Paso 1:[/] Cargando datos desde CSV...\n")
    df = load_raw()

    console.print()
    console.rule("[bold yellow]Visualizacion de Datos[/]")
    console.print()

    # Paso 2: Mostrar resumen
    mostrar_resumen(df)
    console.print()

    # Paso 3: Mostrar muestra
    mostrar_muestra(df)
    console.print()

    console.rule("[bold green]Carga completada exitosamente[/]")
    console.print()

    return df


if __name__ == "__main__":
    main()
