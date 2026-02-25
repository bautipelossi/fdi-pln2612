"""Funciones auxiliares de estrategia del agente."""

from __future__ import annotations

import time
from typing import Any

from .butler_api import (
    BAD_DEST_UNTIL,
    borrar_mail,
    enviar_carta,
    enviar_paquete,
    get_alias,
)
from .config import (
    ALLOW_BREAK_OBJECTIVE,
    CLEAN_INBOX,
    OFFER_COOLDOWN_GLOBAL,
    OFFER_COOLDOWN_PER_DEST,
    log,
)
from .models import InfoPuesto
from .protocol import parse_accept_from_text

# =========================================================
# Estado anti-spam
# =========================================================

LAST_OFFER_TS_DEST: dict[str, float] = {}
LAST_OFFER_TS_GLOBAL: float = 0.0


# =========================================================
# Heurísticas de análisis de recursos
# =========================================================


def faltantes(estado: InfoPuesto) -> dict[str, int]:
    """Calcula los recursos que faltan para cumplir el objetivo."""
    f = {}
    for r, obj in (estado.Objetivo or {}).items():
        cur = (estado.Recursos or {}).get(r, 0)
        if cur < obj:
            f[r] = obj - cur
    return f


def excedentes(estado: InfoPuesto) -> dict[str, int]:
    """Calcula los recursos que sobran del objetivo."""
    exc = {}
    for r, qty in (estado.Recursos or {}).items():
        if r == "oro":
            continue
        obj = (estado.Objetivo or {}).get(r, 0)
        if qty > obj:
            exc[r] = qty - obj
        if r not in (estado.Objetivo or {}) and qty > 0:
            exc[r] = max(exc.get(r, 0), qty)
    return exc


def can_give(estado: InfoPuesto, item: str, qty: int) -> bool:
    """Verifica si el agente puede dar un recurso sin romper su objetivo."""
    have = estado.Recursos.get(item, 0)
    if have < qty:
        return False
    if ALLOW_BREAK_OBJECTIVE:
        return True
    # modo clase: no romper objetivo
    obj = estado.Objetivo.get(item, 0)
    return (have - obj) >= qty


def objetivo_cumplido(estado: InfoPuesto) -> bool:
    """Verifica si el agente ya cumplió su objetivo."""
    for recurso, cantidad_objetivo in estado.Objetivo.items():
        if estado.Recursos.get(recurso, 0) < cantidad_objetivo:
            return False
    return True


# =========================================================
# Anti-spam
# =========================================================


def can_send_offer_now(dest: str) -> bool:
    """Verifica si se puede enviar una oferta a un destino."""
    global LAST_OFFER_TS_GLOBAL
    now = time.time()
    if dest in BAD_DEST_UNTIL and now < BAD_DEST_UNTIL[dest]:
        return False
    if now - LAST_OFFER_TS_GLOBAL < OFFER_COOLDOWN_GLOBAL:
        return False
    last_d = LAST_OFFER_TS_DEST.get(dest, 0.0)
    if now - last_d < OFFER_COOLDOWN_PER_DEST:
        return False
    return True


def mark_offer_sent(dest: str) -> None:
    """Marca que se envió una oferta a un destino."""
    global LAST_OFFER_TS_GLOBAL
    now = time.time()
    LAST_OFFER_TS_GLOBAL = now
    LAST_OFFER_TS_DEST[dest] = now


# =========================================================
# Procesamiento automático de mails
# =========================================================


def procesar_mails_automaticos(
    mails_by_id: dict[str, dict[str, Any]],
    estado_recursos: dict[str, int],
) -> None:
    """Procesa automáticamente los mails de aceptación."""
    alias = get_alias()

    for mid, mail in list(mails_by_id.items()):
        remi = (mail.get("remi") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()

        if remi.lower() == "sistema":
            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)
            continue

        acc = parse_accept_from_text(cuerpo)
        if acc and remi:
            _, espero = acc
            ok_send = True
            for k, v in espero.items():
                if estado_recursos.get(k, 0) < int(v):
                    ok_send = False
                    break

            if ok_send:
                log.info(f"📨 Cumpliendo aceptación de {remi}: enviando {espero}")
                enviar_paquete(remi, {k: int(v) for k, v in espero.items()})
                enviar_carta(remi, "Envío", f"Listo. Te envié {espero}.")
            else:
                enviar_carta(
                    remi, "No puedo", f"No tengo suficiente para enviarte {espero}."
                )

            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)
