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
