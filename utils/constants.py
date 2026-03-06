"""Constantes de dominio del pipeline — evitan strings hardcodeados dispersos.

Uso:
    from utils.constants import Sentiment, DisconnectedReason

    if sentiment == Sentiment.NEGATIVE:
        ...
"""


class Sentiment:
    """Etiquetas de sentimiento del dominio."""
    POSITIVE = "positivo"
    NEUTRAL  = "neutral"
    NEGATIVE = "negativo"
    ALL      = (POSITIVE, NEUTRAL, NEGATIVE)


class DisconnectedReason:
    """Razones de desconexión presentes en disconnected_reason."""
    AGENT_HANGUP       = "agent_hangup"
    USER_HANGUP        = "user_hangup"
    INACTIVITY         = "inactivity"
    IVR_REACHED        = "ivr_reached"
    MAX_DURATION       = "max_duration_reached"
    SYSTEM_ERROR       = "system_error"
    ALL_ORDERED = (
        AGENT_HANGUP, USER_HANGUP, INACTIVITY,
        IVR_REACHED, MAX_DURATION, SYSTEM_ERROR,
    )
