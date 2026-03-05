"""
main.py - Punto de entrada del pipeline de analisis de llamadas.
Usa load_data para cargar el CSV y muestra los datos de forma agradable.
"""

import os
import sys
from pathlib import Path

# Forzar UTF-8 en Windows para evitar errores de codificacion
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Agregar scripts/ y utils/ al path para importar módulos del pipeline
base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path / "scripts"))
sys.path.insert(0, str(base_path / "utils"))

from rich.console import Console

from load_data import load_raw
from clean_data import clean
from display import mostrar_resumen, mostrar_muestra, mostrar_columnas_nuevas


console = Console(force_terminal=True, safe_box=True)


def main(verbose: bool = True) -> "pl.DataFrame":
    """
    Ejecuta el pipeline completo.
    verbose=True  → imprime tablas rich en consola (uso desde main.py).
    verbose=False → corre el pipeline sin output (uso desde notebooks).
    """
    if verbose:
        console.print()
        console.rule("[bold yellow]Pipeline de Analisis de Llamadas[/]")
        console.print()

    # Paso 1: Cargar datos
    if verbose:
        console.print("[bold]>> Paso 1:[/] Cargando datos desde CSV...\n")
    df = load_raw()

    # Paso 2: Limpiar y normalizar
    if verbose:
        console.print("\n[bold]>> Paso 2:[/] Limpiando y normalizando datos...\n")
    df_clean = clean(df)

    if verbose:
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
    main(verbose=True)
