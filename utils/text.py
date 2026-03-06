"""Utilidades de normalización de texto en español."""

import re

_ACCENTS_TABLE = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
_RE_PUNCT = re.compile(r"[^\w\s]")


def strip_accents(text: str) -> str:
    """Remueve tildes y ñ → n, preserva mayúsculas/minúsculas."""
    return text.translate(_ACCENTS_TABLE)


def normalize_text(text: str) -> str:
    """Lowercase, sin tildes, puntuación → espacios, strip."""
    return _RE_PUNCT.sub(" ", text.lower().translate(_ACCENTS_TABLE)).strip()


def extract_transcript_lines(transcript: str | None, speaker: str) -> list[str]:
    """Extrae utterances de un speaker del transcript.

    Args:
        transcript: texto del transcript con líneas formato "Speaker: texto".
        speaker: nombre del hablante a filtrar, p.ej. "User" o "Agent".

    Returns:
        Lista de strings con el texto de cada turno del speaker (sin el prefijo).
        Retorna [] si transcript es None o vacío.
    """
    if not transcript:
        return []
    prefix = f"{speaker}:"
    return [
        line.strip()[len(prefix):].strip()
        for line in transcript.split("\n")
        if line.strip().startswith(prefix)
    ]
