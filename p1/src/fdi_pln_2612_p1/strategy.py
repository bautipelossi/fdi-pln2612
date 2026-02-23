"""Estrategias y heurísticas de decisión del agente."""

from __future__ import annotations

import random
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
    ACCEPT_ANY,
    ALLOW_BREAK_OBJECTIVE,
    CLEAN_INBOX,
    MAX_MAILS,
    OFFER_COOLDOWN_GLOBAL,
    OFFER_COOLDOWN_PER_DEST,
)
from .models import Decision, InfoPuesto
from .protocol import extract_first_item, parse_accept_from_text, parse_offer_from_text

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
                enviar_paquete(remi, {k: int(v) for k, v in espero.items()})
                enviar_carta(remi, "Envío", f"Listo. Te envié {espero}.")
            else:
                enviar_carta(
                    remi, "No puedo", f"No tengo suficiente para enviarte {espero}."
                )

            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)


# =========================================================
# Decisión heurística (fallback)
# =========================================================


def decidir_fallback(
    estado: InfoPuesto, gente: list[str], mails_by_id: dict[str, dict[str, Any]]
) -> Decision:
    """Heurísticas de respaldo cuando el LLM no está disponible."""
    alias = get_alias()
    f = faltantes(estado)
    exc = excedentes(estado)

    # limitar crecimiento de buzón
    if CLEAN_INBOX and len(mails_by_id) > MAX_MAILS:
        for mid in list(mails_by_id.keys())[MAX_MAILS:]:
            borrar_mail(mid)
            mails_by_id.pop(mid, None)

    # 1) aceptar ofertas entrantes
    for mid, mail in list(mails_by_id.items()):
        remi = (mail.get("remi") or "").strip()
        cuerpo = (mail.get("cuerpo") or "").strip()
        if not remi or remi == alias:
            continue

        parsed = parse_offer_from_text(cuerpo)
        if not parsed:
            continue

        quiero, ofrezco = parsed
        need_pair = extract_first_item(quiero)
        offer_pair = extract_first_item(ofrezco)
        if not need_pair or not offer_pair:
            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)
            continue
        need_item, need_qty = need_pair
        offer_item, offer_qty = offer_pair
        need_qty = int(need_qty)
        offer_qty = int(offer_qty)

        if not can_give(estado, need_item, need_qty):
            if CLEAN_INBOX:
                borrar_mail(mid)
                mails_by_id.pop(mid, None)
            continue

        if ACCEPT_ANY or f.get(offer_item, 0) > 0:
            return Decision(
                razonamiento=f"Acepto: puedo dar {need_qty} {need_item}; recibiría {offer_qty} {offer_item}.",
                accion={"tipo": "aceptar", "mensaje_id": mid},
            )

        if CLEAN_INBOX:
            borrar_mail(mid)
            mails_by_id.pop(mid, None)

    # 2) ofertar proactivo
    otros = [a for a in gente if a != alias]
    if not otros:
        return Decision(
            razonamiento="No hay otros agentes conectados.", accion={"tipo": "esperar"}
        )

    random.shuffle(otros)
    dest = None
    for cand in otros:
        if can_send_offer_now(cand):
            dest = cand
            break
    if dest is None:
        return Decision(
            razonamiento="Cooldown/global o destinos malos -> espero.",
            accion={"tipo": "esperar"},
        )

    # elegir qué pedir y qué ofrecer
    if f:
        need_item = max(f, key=lambda k: f[k])
    else:
        need_item = random.choice(list(estado.Recursos.keys()) or ["madera"])

    offer_item = None
    if exc:
        offer_item = max(exc, key=lambda k: exc[k])
    else:
        cand_items = [
            k for k, v in estado.Recursos.items() if v >= 1 and can_give(estado, k, 1)
        ]
        if cand_items:
            offer_item = random.choice(cand_items)

    if not offer_item or offer_item == need_item:
        return Decision(
            razonamiento="No tengo buen recurso para ofrecer ahora.",
            accion={"tipo": "esperar"},
        )

    return Decision(
        razonamiento=f"Oferto a {dest}: pido {need_item} y ofrezco {offer_item}.",
        accion={
            "tipo": "ofertar",
            "dest": dest,
            "need_recurso": need_item,
            "need_cantidad": 1,
            "offer_recurso": offer_item,
            "offer_cantidad": 1,
        },
    )
