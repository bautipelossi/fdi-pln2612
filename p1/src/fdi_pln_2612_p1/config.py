"""Configuración del agente desde variables de entorno."""

from __future__ import annotations

import logging
import os

# =========================================================
# LOGGING
# =========================================================

# Silenciar librerías ruidosas (HTTP connections, etc.)
for _lib in ("urllib3", "httpcore", "httpx", "requests"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fdi_pln_agent")


# =========================================================
# CONFIG / ENTORNO
# =========================================================

BUTLER_URL = os.getenv("FDI_PLN__BUTLER_ADDRESS")
ALIAS = os.getenv("FDI_PLN__ALIAS", "fdi-pln-2612")

SLEEP_SECONDS = int(os.getenv("FDI_PLN__SLEEP_SECONDS", "6"))
REQUEST_TIMEOUT = float(os.getenv("FDI_PLN__TIMEOUT", "10"))

# Reducir carga
GENTE_EVERY = int(os.getenv("FDI_PLN__GENTE_EVERY", "3"))  # /gente cada N ciclos

# Anti-spam
OFFER_COOLDOWN_PER_DEST = int(os.getenv("FDI_PLN__OFFER_COOLDOWN", "25"))
OFFER_COOLDOWN_GLOBAL = int(os.getenv("FDI_PLN__OFFER_COOLDOWN_GLOBAL", "5"))
BAD_DEST_SECONDS = int(os.getenv("FDI_PLN__BAD_DEST_SECONDS", "60"))

# TEST MODE (casa): permitir trade aunque baje tu objetivo (si tenés el recurso)
ALLOW_BREAK_OBJECTIVE = os.getenv("FDI_PLN__ALLOW_BREAK_OBJECTIVE", "0") == "1"
# Limpiar buzón (para que no crezca infinito)
CLEAN_INBOX = os.getenv("FDI_PLN__CLEAN_INBOX", "1") == "1"

DEBUG = os.getenv("FDI_PLN__DEBUG", "1") == "1"

# LLM
LLM_MODEL = os.getenv("FDI_PLN__LLM_MODEL", "llama3.2:3b")
USE_LLM = os.getenv("FDI_PLN__USE_LLM", "1") == "1"
LLM_TIMEOUT = float(os.getenv("FDI_PLN__LLM_TIMEOUT", "45"))
LLM_TEMPERATURE = float(os.getenv("FDI_PLN__LLM_TEMPERATURE", "0.3"))
LLM_NUM_PREDICT = int(os.getenv("FDI_PLN__LLM_NUM_PREDICT", "80"))

# Headers HTTP
HEADERS = {"Connection": "close"}  # evita keep-alive y baja 10053/10054


def validate_config() -> None:
    """Valida que la configuración mínima esté presente."""
    if not BUTLER_URL:
        raise RuntimeError(
            "FDI_PLN__BUTLER_ADDRESS no está definida (ej: http://127.0.0.1:8000)"
        )
