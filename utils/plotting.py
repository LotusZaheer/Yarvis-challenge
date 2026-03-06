"""Utilidades compartidas de matplotlib para el pipeline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 150


def savefig(fig, out_path: Path, report: bool = False) -> None:
    """Guarda, cierra y opcionalmente reporta una figura matplotlib."""
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    if report:
        print(f"[INFO] Figura: {out_path.name}")
