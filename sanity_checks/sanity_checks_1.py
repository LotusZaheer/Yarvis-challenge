import polars as pl
import logging
from datetime import datetime
from pathlib import Path
from rich.logging import RichHandler

from rich.console import Console
from rich.table import Table


# ---------------------------------------------------------------------------
# Configuración base
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path().resolve()
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "data_calls.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
LOGS_DIR = PROJECT_ROOT / "logs"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
script_name = Path(__file__).stem
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"{script_name}_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
logger.info("Iniciando carga de datos")

df = pl.read_csv(
    RAW_CSV,
    separator=",",
    null_values=["NULL", "NaN", "nan", ""]
)

COLUMN_RENAME = {
    "Campaign Id": "campaign_id",
    "Name": "name",
    "Target Id": "target_id",
    "Call URL": "call_url",
    "Connected": "connected",
    "Disconnected Reason": "disconnected_reason",
    "Duration (ms)": "duration_ms",
    "Transcript": "transcript",
    "Post Call Analysis": "post_call_analysis",
    "Executed At": "executed_at",
}

df = df.rename(COLUMN_RENAME)

df = df.with_columns(
    pl.col("call_url").is_not_null().alias("call_completed")
)

logger.info(f"Dataset cargado correctamente | filas: {df.height}")

# ---------------------------------------------------------------------------
# Columnas base
# ---------------------------------------------------------------------------
cols = [
    "campaign_id",
    "name",
    "target_id",
    "call_url",
    "connected",
    "disconnected_reason",
    "post_call_analysis",
    "call_completed"
]

# ---------------------------------------------------------------------------
# Validación igualdad total
# ---------------------------------------------------------------------------
equals_flag = df["call_completed"].equals(df["connected"])
logger.info("--------------------------------------------------")
logger.info(f"¿connected == call_completed?: {equals_flag}")
logger.info("--------------------------------------------------")

# ---------------------------------------------------------------------------
# Función combinaciones
# ---------------------------------------------------------------------------
console = Console()

def sample_combo(df: pl.DataFrame, connected_val: bool, completed_val: bool) -> pl.DataFrame:
    subset = df.filter(
        (pl.col("connected") == connected_val) &
        (pl.col("call_completed") == completed_val)
    ).select(cols)

    logger.info(
        f"Combinación | connected={connected_val} "
        f"| call_completed={completed_val} "
        f"| total={subset.height}"
    )

    if subset.height == 0:
        return subset

    subset_sample = subset.sample(n=min(5, subset.height))

    # Tabla solo visual (no log)
    table = Table(title=f"connected={connected_val} | call_completed={completed_val}", show_lines=False)

    for col in subset_sample.columns:
        table.add_column(col)

    for row in subset_sample.rows():
        table.add_row(*[str(x) for x in row])

    console.print(table)
    console.print()

    return subset_sample

# ---------------------------------------------------------------------------
# 4 combinaciones
# ---------------------------------------------------------------------------
df_tt = sample_combo(df, True, True)
df_tf = sample_combo(df, True, False)
df_ft = sample_combo(df, False, True)
df_ff = sample_combo(df, False, False)

df_combinations = pl.concat([df_tt, df_tf, df_ft, df_ff])

df_combinations.write_csv(
    INTERIM_DIR / "muestra_4_combinaciones.csv"
)

logger.info("--------------------------------------------------")
logger.info("Conteo por combinación")
logger.info("--------------------------------------------------")

logger.info(f"TT (True,  True ) : {df_tt.height}")
logger.info(f"TF (True,  False) : {df_tf.height}")

if df_ft.height > 0:
    logger.error(
        f"FT (False, True ) : {df_ft.height} <-- INCONSISTENCIA LOGICA"
    )
else:
    logger.info(f"FT (False, True ) : {df_ft.height}")

logger.info(f"FF (False, False) : {df_ff.height}")
logger.info("--------------------------------------------------")

# ---------------------------------------------------------------------------
# Diferencias
# ---------------------------------------------------------------------------
df_diff = df.filter(
    pl.col("call_completed") != pl.col("connected")
)

df_connected_false = df.filter(
    pl.col("connected") == False
)

df_diff_sample = (
    df_diff.select(cols).sample(n=min(10, df_diff.height))
    if df_diff.height > 0 else df_diff
)

df_false_sample = (
    df_connected_false.select(cols).sample(n=min(10, df_connected_false.height))
    if df_connected_false.height > 0 else df_connected_false
)

df_final = pl.concat([df_diff_sample, df_false_sample])

df_final.write_csv(
    INTERIM_DIR / "muestra_10_diff_y_10_false.csv"
)

logger.info("--------------------------------------------------")
logger.info("Resumen validación diferencias")
logger.info("--------------------------------------------------")
logger.info(f"Filas donde connected != call_completed : {df_diff.height}")
logger.info(f"Filas donde connected == False          : {df_connected_false.height}")

if df_diff.height > 0:
    logger.warning("Existen inconsistencias entre connected y call_completed")

logger.info("Proceso finalizado correctamente")
logger.info("--------------------------------------------------")