"""Funciones para interactuar con la API de Butler."""

from __future__ import annotations

import random
import time
from typing import Any

from .config import ALIAS, BAD_DEST_SECONDS, log
from .http_client import http_delete, http_get, http_post
from .models import InfoPuesto

# Estado mutable del módulo
_current_alias: str = ALIAS
BAD_DEST_UNTIL: dict[str, float] = {}


def get_alias() -> str:
    """Retorna el alias actual del agente."""
    return _current_alias


def set_alias(new_alias: str) -> None:
    """Establece el alias del agente."""
    global _current_alias
    _current_alias = new_alias


def set_alias_in_butler() -> str:
    """Registra el alias del agente en Butler."""
    global _current_alias
    base = ALIAS

    for attempt in range(1, 6):
        candidate = base if attempt == 1 else f"{base}-{attempt}"
        r = http_post(f"/alias/{candidate}", payload={})
        if r.status_code == 200:
            log.info(f"Alias fijado en Butler: {candidate}")
            _current_alias = candidate
            return candidate
        if r.status_code == 403:
            continue
        if DEBUG:
            log.warning(f"No pude setear alias {candidate}: {r.status_code} {r.text}")

    candidate = f"{base}-{random.randint(1000, 9999)}"
    r = http_post(f"/alias/{candidate}", payload={})
    if r.status_code == 200:
        log.info(f"Alias fijado en Butler: {candidate}")
        _current_alias = candidate
        return candidate

    log.warning("No pude setear alias final, sigo con ALIAS base.")
    _current_alias = base
    return base


def get_info() -> InfoPuesto:
    """Obtiene el estado del puesto del agente."""
    r = http_get("/info", params={"agente": _current_alias})
    r.raise_for_status()
    return InfoPuesto.model_validate(r.json())


def get_gente() -> list[str]:
    """Obtiene la lista de agentes conectados."""
    r = http_get("/gente")
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data:
        if isinstance(item, dict) and "alias" in item:
            out.append(item["alias"])
        elif isinstance(item, str):
            out.append(item)
    return out


def enviar_carta(dest: str, asunto: str, cuerpo: str) -> bool:
    """Envía una carta a otro agente."""
    payload = {"remi": _current_alias, "dest": dest, "asunto": asunto, "cuerpo": cuerpo}
    try:
        r = http_post("/carta", payload=payload)
    except Exception as e:
        log.warning(f"[CARTA] {dest} error: {e}")
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
        return False

    ok = r.status_code == 200
    if not ok:
        log.warning(f"[CARTA] {dest} falló ({r.status_code})")
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
    return ok


def borrar_mail(mail_id: str) -> None:
    """Borra un mail del buzón."""
    http_delete(f"/mail/{mail_id}", params={"agente": _current_alias})


def enviar_paquete(dest: str, paquete: dict[str, int]) -> bool:
    """Envía un paquete de recursos a otro agente."""
    r = http_post(
        f"/paquete/{dest}", payload=paquete, params={"agente": _current_alias}
    )
    ok = r.status_code == 200
    if not ok:
        log.warning(f"[PAQUETE] {dest} falló ({r.status_code})")
        BAD_DEST_UNTIL[dest] = time.time() + BAD_DEST_SECONDS
    return ok
