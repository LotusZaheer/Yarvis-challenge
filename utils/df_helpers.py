"""Helpers de DataFrame de uso común en el pipeline.

Uso:
    from utils.df_helpers import connected_calls

    df_conn = connected_calls(df)
"""

import polars as pl


def connected_calls(df: pl.DataFrame) -> pl.DataFrame:
    """Retorna solo las filas donde connected == True.

    Reemplaza el patrón repetido:
        df.filter(pl.col("connected") == True)
    por una función con nombre explícito y semánticamente correcta.
    """
    return df.filter(pl.col("connected"))
