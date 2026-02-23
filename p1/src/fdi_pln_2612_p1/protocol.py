"""Protocolo de mensajes para ofertas y aceptaciones."""

from __future__ import annotations

import json
import re

from .config import DEBUG, log

# =========================================================
# Expresiones regulares para parsear mensajes
# =========================================================

TAG_OFFER_RE = re.compile(
    r"\[OFERTA_V1\]\s*quiero=(\{.*?\})\s*ofrezco=(\{.*?\})",
    re.IGNORECASE,
)
TAG_ACCEPT_RE = re.compile(
    r"\[ACEPTO_V1\]\s*te_envio=(\{.*?\})\s*espero=(\{.*?\})",
    re.IGNORECASE,
)

# Fallback para mensajes en lenguaje natural
OFFER_RE = re.compile(
    r"necesit[oa]\s+(\d+)\s+([a-zA-ZáéíóúñÑ]+).*?ofrezc[oa]\s+(\d+)\s+([a-zA-ZáéíóúñÑ]+)",
    re.IGNORECASE | re.DOTALL,
)


# =========================================================
# Builders
# =========================================================


def build_offer_body(
    need_item: str, need_qty: int, offer_item: str, offer_qty: int
) -> str:
    """Construye el cuerpo de una carta de oferta."""
    line1 = f"[OFERTA_V1] quiero={json.dumps({need_item: need_qty}, ensure_ascii=False)} ofrezco={json.dumps({offer_item: offer_qty}, ensure_ascii=False)}"
    line2 = f"Necesito {need_qty} {need_item} y ofrezco {offer_qty} {offer_item}."
    return line1 + "\n" + line2


def build_accept_body(
    give_to_other: dict[str, int], expect_from_other: dict[str, int]
) -> str:
    """Construye el cuerpo de una carta de aceptación."""
    line1 = f"[ACEPTO_V1] te_envio={json.dumps(give_to_other, ensure_ascii=False)} espero={json.dumps(expect_from_other, ensure_ascii=False)}"
    # Tomar primer item de cada dict para el texto legible
    give_items = list(give_to_other.items())
    expect_items = list(expect_from_other.items())
    gk, gv = give_items[0] if give_items else ("", 0)
    ek, ev = expect_items[0] if expect_items else ("", 0)
    line2 = f"Acepto tu oferta. Te envié {gv} {gk}. Enviame {ev} {ek}."
    return line1 + "\n" + line2


# =========================================================
# Parsers
# =========================================================


def extract_first_item(d: dict[str, int]) -> tuple[str, int] | None:
    """Extrae el primer item de un dict. Retorna None si está vacío o tiene >1 item."""
    items = list(d.items())
    if len(items) != 1:
        if DEBUG and len(items) > 1:
            log.warning(f"Oferta con múltiples items ({len(items)}), ignorando.")
        return None
    return items[0]


def parse_offer_from_text(texto: str) -> tuple[dict[str, int], dict[str, int]] | None:
    """Parsea una oferta desde el texto de un mensaje."""
    t = texto or ""
    m = TAG_OFFER_RE.search(t)
    if m:
        try:
            quiero = json.loads(m.group(1))
            ofrezco = json.loads(m.group(2))
            if isinstance(quiero, dict) and isinstance(ofrezco, dict):
                return quiero, ofrezco
        except Exception:
            pass

    m2 = OFFER_RE.search(t)
    if m2:
        need_qty = int(m2.group(1))
        need_item = m2.group(2).lower()
        offer_qty = int(m2.group(3))
        offer_item = m2.group(4).lower()
        return {need_item: need_qty}, {offer_item: offer_qty}

    return None


def parse_accept_from_text(texto: str) -> tuple[dict[str, int], dict[str, int]] | None:
    """Parsea una aceptación desde el texto de un mensaje."""
    t = texto or ""
    m = TAG_ACCEPT_RE.search(t)
    if not m:
        return None
    try:
        te_envio = json.loads(m.group(1))
        espero = json.loads(m.group(2))
        if isinstance(te_envio, dict) and isinstance(espero, dict):
            return te_envio, espero
    except Exception:
        return None
    return None
