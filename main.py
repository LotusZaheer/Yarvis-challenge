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
from sentiment_analysis import analyze_sentiment
from contactability import analyze_contactability
from cluster_contacts import cluster_contacts
from agent_performance import analyze_agent_performance
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

    # Paso 3: Análisis de sentimiento propio (Tarea 2)
    if verbose:
        console.print("\n[bold]>> Paso 3:[/] Analizando sentimiento de transcripts...\n")
    df_clean = analyze_sentiment(df_clean)

    # Paso 4: Contactabilidad — patrones de conexión (Tarea 0)
    if verbose:
        console.print("\n[bold]>> Paso 4:[/] Analizando contactabilidad...\n")
    df_clean = analyze_contactability(df_clean)

    # Paso 5: Clustering de contactos conectados (Tarea 1)
    if verbose:
        console.print("\n[bold]>> Paso 5:[/] Clustering de llamadas conectadas...\n")
    df_clean = cluster_contacts(df_clean)

    # Paso 6: Desempeño del agente (Tarea 3)
    if verbose:
        console.print("\n[bold]>> Paso 6:[/] Analizando desempeno del agente...\n")
    df_clean = analyze_agent_performance(df_clean)

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

        console.rule("[bold green]Pipeline completado exitosamente[/]")
        console.print()

    return df_clean


if __name__ == "__main__":
    main(verbose=True)
