"""
Script 04 - Análisis de sentimiento propio (Tarea 2).
Enfoque léxico en español sobre texto de usuario extraído del transcript.
Entrada : pl.DataFrame limpio (con columna transcript_text)
Salida  : pl.DataFrame con columna sentiment_own (negativo/neutral/positivo)
"""

import re
import sys
from pathlib import Path

import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "calls_clean.csv"
SENTIMENT_CACHE = PROJECT_ROOT / "data" / "processed" / "cache_sentiment.csv"

# ---------------------------------------------------------------------------
# Léxico español para contexto de atención al cliente
# ---------------------------------------------------------------------------
_POSITIVE = {
    "excelente", "perfecto", "genial", "maravilloso", "increible", "encantado",
    "feliz", "satisfecho", "contento", "gracias", "buenisimo", "estupendo",
    "solucionado", "resuelto", "funcionando", "funciona", "ayudo", "ayuda",
    "adelante", "dale", "interesa", "quiero", "gustaria", "me encanta",
    "sin problema", "listo", "bien", "buena", "bueno", "ok", "vale",
    "amable", "rapido", "eficiente", "tranquilo", "conforme", "entendido",
    "claro que si", "de acuerdo", "me gustaria", "si me interesa",
    "activen", "activar", "activo", "renovar", "recargar", "pagar",
    "probarlo", "interesante", "excelentes", "perfecta", "felices",
}

_NEGATIVE = {
    "mal", "malo", "mala", "terrible", "pesimo", "horrible", "desastre",
    "problema", "falla", "fallo", "error", "queja", "reclamo", "reclamos",
    "molesto", "frustrado", "decepcionado", "harto", "cansado", "enojado",
    "molestia", "inconveniente", "nunca", "imposible", "mentira",
    "incumplimiento", "lento", "muy lento", "cortado", "no funciona",
    "no sirve", "se cayo", "se cae", "perdida", "cancelar", "cancelacion",
    "baja", "odio", "asco", "inaceptable", "inactivo", "inactiva",
    "no puedo", "dificil", "complicado", "complicada", "confundido",
    "no entiendo", "fallo", "fallando", "no conecta", "sin servicio",
    "sin internet", "sin señal", "no pasa", "no pueden", "no resuelven",
    "cobro", "cobros", "cobran", "cobraron", "mal cobrado", "robo",
    "estafa", "fraude", "engano", "mentiras", "desconectado",
}

# Frases específicas con mayor peso (se evalúan antes del léxico individual)
_POSITIVE_PHRASES = [
    "si me interesa", "claro que si", "de acuerdo", "me gustaria probarlo",
    "adelante", "activen", "me parece bien", "muchas gracias", "todo bien",
    "sin problema", "ya lo resolvieron", "ya funciona", "quiero el servicio",
]

_NEGATIVE_PHRASES = [
    "no me interesa", "no gracias", "no necesito", "no quiero",
    "quiero cancelar", "quiero darme de baja", "es un problema",
    "no funciona nada", "no sirve nada", "muy mal servicio",
    "pesimo servicio", "terrible servicio", "ya me voy a ir",
    "me voy a cambiar", "me tienen harto", "no me cobren mas",
]

_RE_PUNCT = re.compile(r"[^\w\s]")
_RE_ACCENTS = str.maketrans("áéíóúüñ", "aeiooun")


def _normalize(text: str) -> str:
    text = text.lower().translate(_RE_ACCENTS)
    text = _RE_PUNCT.sub(" ", text)
    return text


def _extract_user_text(transcript: str) -> str:
    """Extrae únicamente las líneas del usuario del transcript."""
    if not transcript:
        return ""
    lines = []
    for line in transcript.split("\n"):
        stripped = line.strip()
        if stripped.startswith("User:"):
            lines.append(stripped[5:].strip())
    return " ".join(lines)


def _score_text(text: str) -> float:
    """Devuelve score de sentimiento: positivo > 0, negativo < 0."""
    if not text or len(text.strip()) < 3:
        return 0.0

    normalized = _normalize(text)

    # Frases (mayor peso = 2)
    pos_score = sum(2 for p in _POSITIVE_PHRASES if p in normalized)
    neg_score = sum(2 for p in _NEGATIVE_PHRASES if p in normalized)

    # Palabras individuales
    tokens = set(normalized.split())
    pos_score += sum(1 for w in tokens if w in _POSITIVE)
    neg_score += sum(1 for w in tokens if w in _NEGATIVE)

    return float(pos_score - neg_score)


def _classify(score: float, user_text: str) -> str:
    """Convierte score numérico en etiqueta de sentimiento."""
    if not user_text or len(user_text.strip()) < 5:
        return "neutral"
    if score >= 1.0:
        return "positivo"
    if score <= -1.0:
        return "negativo"
    return "neutral"


def analyze_sentiment(df: pl.DataFrame) -> pl.DataFrame:
    """
    Añade columna sentiment_own al DataFrame.
    Si existe cache_sentiment.csv, lo carga y hace join por call_url.
    Retorna df con la nueva columna.
    """
    if SENTIMENT_CACHE.exists():
        cached = pl.read_csv(SENTIMENT_CACHE)
        df = df.join(cached.select(["call_url", "sentiment_own"]), on="call_url", how="left")
        df = df.with_columns(pl.col("sentiment_own").fill_null("neutral"))
        print(f"[INFO] Cache encontrado: {SENTIMENT_CACHE.name}")
        return df

    transcripts = df["transcript_text"].to_list()

    sentiments = []
    for txt in transcripts:
        if txt is None:
            sentiments.append("neutral")
            continue
        user_text = _extract_user_text(txt)
        score = _score_text(user_text)
        sentiments.append(_classify(score, user_text))

    df = df.with_columns(pl.Series("sentiment_own", sentiments, dtype=pl.Utf8))

    # Guardar checkpoint (call_url + sentiment_own)
    df.select(["call_url", "sentiment_own"]).write_csv(SENTIMENT_CACHE)

    # Distribución propia
    dist = df["sentiment_own"].value_counts(sort=True)
    counts = {row[0]: row[1] for row in dist.iter_rows()}
    print(
        f"[INFO] sentiment_own: positivo={counts.get('positivo', 0):,}  "
        f"neutral={counts.get('neutral', 0):,}  "
        f"negativo={counts.get('negativo', 0):,}"
    )

    # Comparar con pca_sentimiento en registros donde ambos existen
    overlap = df.filter(pl.col("pca_sentimiento").is_not_null())
    if overlap.height > 0:
        match = overlap.filter(pl.col("sentiment_own") == pl.col("pca_sentimiento")).height
        pct_match = match / overlap.height
        print(f"[INFO] Concordancia con pca_sentimiento ({overlap.height} registros): {pct_match:.1%}")

    return df


if __name__ == "__main__":
    df = pl.read_csv(CLEAN_CSV)
    df = analyze_sentiment(df)
    print(df.select(["target_id", "pca_sentimiento", "sentiment_own"]).head(10))
